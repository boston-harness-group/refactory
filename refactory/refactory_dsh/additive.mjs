// Declare a normal web + Refactory profile. No source profile is copied.
import { readFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { loadAtomicWrite } from './profile_manifest.mjs';

export async function ensureAdditiveProfile(anchor) {
  const root = createRequire(join(anchor, 'package.json'));
  const installAnchor = root.resolve('@deepseek-ai/dsh/package.json');
  const harness = createRequire(installAnchor);
  const boot = await import(pathToFileURL(harness.resolve('@deepseek-ai/dsh-app-boot')).href);
  const { resolveDshHome } = await import(pathToFileURL(harness.resolve('@deepseek-ai/dsh-home-paths')).href);
  const { parseDocument } = harness('yaml');
  const directory = boot.resolveProfileDir('refactory-web', resolveDshHome());
  const atomic = await loadAtomicWrite(anchor);
  const manifestPath = join(directory, 'package.json');
  boot.initProfile(directory, boot.PROFILE_TEMPLATES.web.bundles);
  await atomic.withFileLock(manifestPath, async () => {
    const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
    if (manifest.dsh?.profile?.refactoryAdditive === true) return;
    if (!manifest.dsh?.profile?.bundles?.includes('@deepseek-ai/dsh-web-app')) {
      throw new Error('refactory-web must be a web-based profile');
    }
    const patchPath = join(directory, 'cordis.patch.yml');
    const document = parseDocument(await readFile(patchPath, 'utf8'));
    if (document.errors.length) throw document.errors[0];
    if (document.contents === null) document.contents = parseDocument('[]').contents;
    // The normal Refactory bundles contribute CEO and the Modes panel. Their
    // standalone brand filters presets, so retain the native web picker here.
    const additions = parseDocument('- id: ui-brand-refactory\n  disabled: true\n- id: ui-brand-official\n  disabled: false\n');
    for (const row of additions.contents.items) document.contents.items.push(row);
    await atomic.writeFileAtomic(patchPath, document.toString(), { mode: 0o600 });
    manifest.dsh.profile.refactoryAdditive = true;
    await atomic.writeFileAtomic(manifestPath, JSON.stringify(manifest, null, 2) + '\n', { mode: 0o600 });
  });
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    await ensureAdditiveProfile(process.argv[2]);
  } catch (error) {
    process.stderr.write(`refactory-dsh: ${error.message}\n`);
    process.exitCode = 1;
  }
}
