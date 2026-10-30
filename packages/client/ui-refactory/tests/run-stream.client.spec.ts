/**
 * The hand-written strict codecs carry both harness contract faces.
 *
 * A descriptor mounts into whichever harness serves the profile, and the strict
 * `TypertCodec` changed from `schema` (through `0.1.6-alpha.1`) to a `create()`
 * factory (`e459e32637`, released in `0.1.6-alpha.2`). The pinned harness is
 * `0.2.0-rc.2` and declares only `create`, so a seventh site written as a raw
 * object literal that carried only `schema` would no longer typecheck against
 * the pin. This spec is the guard that every site goes through `strictCodec`.
 *
 * The `schema` assertion is the bridge half, kept for a serving harness older
 * than `0.1.6-alpha.2`: the helper drops `schema`, and this expectation is
 * removed with it, only once no supported serving harness is that old.
 */
import { describe, expect, it } from 'vitest'
import { runStreamContribution, watchFrame, watchStreamContribution } from '../src/client/run-stream.ts'

/** The two faces of one hand-written codec, as either harness generation
 * declares them; the pinned types carry only `create`, so both are optional
 * here. */
interface BothCodecFaces {
  schema?: { parse?: unknown }
  create?: () => { parse?: unknown }
}

describe('run-stream strict codecs', () => {
  it('carries the schema and create faces on every contribution codec', () => {
    const codecs = [runStreamContribution, watchStreamContribution].flatMap(contribution =>
      contribution.descriptors.flatMap(descriptor => [
        ...descriptor.parameters.map(parameter => parameter.codec),
        descriptor.result,
      ]),
    )
    // Five sites today: `runStream`'s request and result, `watchStream`'s run,
    // after and result. A new raw codec joins this list and is checked below.
    expect(codecs.length).toBe(5)
    for (const raw of codecs) {
      const codec = raw as unknown as BothCodecFaces
      expect(typeof codec.schema?.parse).toBe('function')
      expect(typeof codec.create).toBe('function')
      expect(typeof codec.create?.().parse).toBe('function')
    }
  })

  it('accepts a watch frame whose activity carries none of the paging members', () => {
    // T-23a (R-6.2-3): the paging members are optional in the wire type, so a
    // frame from a console that does not page — or from one older than the
    // contract — must still validate. This is a codec-shape assertion only: it
    // mounts no panel behaviour, and the reader that tolerates the absence is
    // the panel's own suite's business (M3).
    const frame = {
      status: { run_id: 'live', root: '/runs/live', state: 'running', cursor: 4 },
      steering: { run: '/runs/live', gates: [] },
    }
    expect(watchFrame(frame)).toMatchObject({ status: { run_id: 'live' }, steering: { run: '/runs/live' } })
    // The same frame with the paging members present is the same frame, and one
    // that is not a watch frame at all is still refused.
    const paged = {
      ...frame,
      status: {
        ...frame.status,
        activity: { cursor: 600, after: 400, total: 269_900, page_size: 200, has_more: true, events: [{}], outputs: {} },
      },
    }
    expect(watchFrame(paged)).toMatchObject({ status: { activity: { cursor: 600, has_more: true } } })
    expect(() => watchFrame({ ...frame, status: { run_id: 'live' } })).toThrow('Invalid durable run frame')
  })
})
