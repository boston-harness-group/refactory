/**
 * The hand-written strict codecs carry both harness contract faces.
 *
 * A descriptor mounts into whichever harness serves the profile, and the strict
 * `TypertCodec` changed from `schema` (through `0.1.6-alpha.1`) to a `create()`
 * factory (`e459e32637`, released in `0.1.6-alpha.2`). The pinned harness does
 * not require `create`, so a seventh site written as a raw object literal still
 * typechecks against the pin and only fails at boot on the other harness. This
 * spec is the guard that every site goes through `strictCodec`.
 *
 * The `schema` assertion is the bridge half: when the pinned harness and the
 * serving harness are both `0.1.6-alpha.2` or later, the helper drops `schema`
 * and this expectation is removed with it.
 */
import { describe, expect, it } from 'vitest'
import { runStreamContribution, watchStreamContribution } from '../src/client/run-stream.ts'

/** The two faces of one hand-written codec, as either harness version declares
 * them; the pinned types carry only `schema`, so both are optional here. */
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
})
