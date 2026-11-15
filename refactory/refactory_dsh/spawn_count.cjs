// Count the processes a Node process starts, when REFACTORY_SPAWN_LOG names a file.
// The Node half of `spawn_count.py`, which puts this file on NODE_OPTIONS; a
// measurement only, and a no-op when the variable is unset.
'use strict';
const log = process.env.REFACTORY_SPAWN_LOG;
if (log) {
  const fs = require('node:fs');
  const childProcess = require('node:child_process');
  const { syncBuiltinESMExports } = require('node:module');
  const write = (line) => {
    try {
      fs.appendFileSync(log, JSON.stringify({ pid: process.pid, runtime: 'node', ...line }) + '\n');
    } catch {
      // A measurement never fails the process it measures.
    }
  };
  const argvOf = (file, args) => [String(file), ...(Array.isArray(args) ? args : [])]
    .slice(0, 6).map((part) => String(part).slice(0, 120));
  write({ kind: 'start', argv: process.argv.slice(0, 4) });
  // Every asynchronous launch goes through ChildProcess.prototype.spawn.
  const spawn = childProcess.ChildProcess.prototype.spawn;
  childProcess.ChildProcess.prototype.spawn = function countedSpawn(options) {
    write({ kind: 'launch', program: options && options.file, argv: argvOf(options && options.file, options && options.args && options.args.slice(1)) });
    return spawn.call(this, options);
  };
  // The synchronous ones call module-internal functions, so each export is wrapped once.
  for (const name of ['spawnSync', 'execSync', 'execFileSync']) {
    const original = childProcess[name];
    childProcess[name] = function countedSync(file, args, ...rest) {
      const program = name === 'execSync' ? '/bin/sh' : file;
      write({ kind: 'launch', program, argv: name === 'execSync' ? [String(file).slice(0, 200)] : argvOf(file, args) });
      return original.call(this, file, args, ...rest);
    };
  }
  syncBuiltinESMExports();
}
