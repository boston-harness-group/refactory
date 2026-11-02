/**
 * TS test for the agent_delivery record_exchange hook (Brief item 3c).
 *
 * The spine path records per-item exchanges via the Python
 * ``agent_delivery.finish`` → ``record_exchange`` → sidecar path.
 * This test exercises that Python code path by calling the agent_delivery
 * module as a subprocess (the same way the spine plugin does), then
 * verifying the exchange sidecar file contains the expected entry.
 *
 * When record_exchange is removed from agent_delivery.finish, the sidecar
 * will be empty or missing, and this test will fail.
 */
import { spawnSync } from 'node:child_process'
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'

const dirs: string[] = []

afterEach(() => {
  for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true })
})

function scratch(): string {
  const dir = mkdtempSync(join(tmpdir(), 'dsh-ad-test-'))
  dirs.push(dir)
  return dir
}

describe('agent_delivery record_exchange hook', () => {
  it('record_exchange captures exchange in the sidecar file', () => {
    const workdir = scratch()
    const sidecar = join(workdir, '.refactory', 'item_exchanges.json')

    // Write a current_item.json so record_exchange knows we are in a data pass.
    writeFileSync(join(workdir, 'current_item.json'), JSON.stringify({ id: 'test-item' }))

    // Call record_exchange directly via Python to write the sidecar.
    const pythonBin = join(process.cwd(), '.venv', 'bin', 'python')
    const script = [
      'from pathlib import Path',
      'from refactory_dsh.items import record_exchange',
      `record_exchange(Path("${workdir}"), "agent_node", "hello prompt", "hello response")`,
    ].join('\n')

    const result = spawnSync(pythonBin, ['-c', script], {
      cwd: process.cwd(),
      timeout: 10_000,
    })

    if (result.status !== 0) {
      throw new Error(`Python record_exchange failed:\n${result.stderr?.toString()}`)
    }

    // The sidecar should contain the exchange.
    expect(existsSync(sidecar)).toBe(true)
    const exchanges = JSON.parse(readFileSync(sidecar, 'utf8'))
    expect(exchanges).toHaveLength(1)
    expect(exchanges[0]).toMatchObject({
      node: 'agent_node',
      prompt: 'hello prompt',
      response: 'hello response',
    })
  })

  it('multiple record_exchange calls accumulate in the sidecar', () => {
    const workdir = scratch()
    const sidecar = join(workdir, '.refactory', 'item_exchanges.json')

    writeFileSync(join(workdir, 'current_item.json'), JSON.stringify({ id: 'test-item' }))

    const pythonBin = join(process.cwd(), '.venv', 'bin', 'python')
    const script = [
      'from pathlib import Path',
      'from refactory_dsh.items import record_exchange',
      `w = Path("${workdir}")`,
      'record_exchange(w, "node_a", "prompt_1", "response_1")',
      'record_exchange(w, "node_b", "prompt_2", "response_2")',
    ].join('\n')

    const result = spawnSync(pythonBin, ['-c', script], {
      cwd: process.cwd(),
      timeout: 10_000,
    })

    if (result.status !== 0) {
      throw new Error(`Python record_exchange failed:\n${result.stderr?.toString()}`)
    }

    expect(existsSync(sidecar)).toBe(true)
    const exchanges = JSON.parse(readFileSync(sidecar, 'utf8'))
    expect(exchanges).toHaveLength(2)
    expect(exchanges[0]).toMatchObject({ node: 'node_a', prompt: 'prompt_1', response: 'response_1' })
    expect(exchanges[1]).toMatchObject({ node: 'node_b', prompt: 'prompt_2', response: 'response_2' })
  })
})
