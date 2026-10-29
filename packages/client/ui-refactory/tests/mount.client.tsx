/**
 * Mount the body over a real store instance and a scripted Remote face.
 *
 * The component reads a handful of its props; the rest of the standard kit is
 * framework-injected and never touched here, so one documented cast keeps the
 * harness to what is actually exercised. The store is driven through its own
 * actions, which is what the face does in production.
 */
import { useSyncExternalStore } from 'react'
import { render } from '@testing-library/react'
import type { RenderResult } from '@testing-library/react'
import { makeTranslate } from './translate.client.ts'
import type { SessionListState } from '@deepseek-ai/dsh-api-session-controller/client'
import type { SessionId } from '@deepseek-ai/dsh-session/types'
import type { TabId } from '@deepseek-ai/dsh-client-ui-dockkit'
import { refactoryFace } from '../src/client/face.ts'
import type { RefactoryInjected } from '../src/client/face.ts'
import { RefactoryBody } from '../src/client/RefactoryBody.tsx'
import type { RefactoryBodyProps } from '../src/client/RefactoryBody.tsx'
import { zh } from '../src/client/locales.ts'
import { NO_SETTINGS, createRefactoryStore } from '../src/client/store.ts'
import type {
  RefactoryCandidateRow,
  RefactoryRunStatus,
  RefactorySteering,
} from '@deepseek-ai/dsh-refactory-console/types'
import type {
  Loadable, RefactoryActivityLevel, RefactoryChoices, RefactoryNotice, RefactorySingleState, RefactoryTabState,
} from '../src/client/store.ts'
import { fakeConsole } from './fake-console.client.ts'
import type { FakeConsole } from './fake-console.client.ts'

export const SESSION = 's-test' as SessionId
export const TAB = 'tab-1' as TabId

/** Test-local selector hook over a framework-neutral store instance. */
function hookOf<T>(inst: { subscribe: (fn: () => void) => () => void; getSnapshot: () => T }) {
  return function useSelector<S>(sel: (s: T) => S): S {
    return sel(useSyncExternalStore(inst.subscribe, inst.getSnapshot))
  }
}

/** A live instance of the picker's store, as the framework would mint one per session. */
export type RefactoryStoreInstance = ReturnType<ReturnType<typeof createRefactoryStore>['create']>

/** What a spec states about the tab before mounting it. */
export interface TabSeed {
  /** Seed nothing, so the body's own mount read is what creates the bucket. */
  readonly bare?: boolean
  /** The dictionary the body draws from; the Chinese one by default. */
  readonly dictionary?: typeof zh
  readonly choices?: Loadable<RefactoryChoices>
  readonly chosen?: string | null
  /** The knob values a person changed, by knob name. */
  readonly knobs?: Readonly<Record<string, string>>
  readonly settings?: Partial<RefactoryTabState['settings']>
  readonly watching?: string | null
  readonly frame?: Loadable<RefactoryRunStatus> | null
  /** The journal history the face would have written; `null` before any page. */
  readonly activity?: { after: number; cursor: number; events: readonly Readonly<Record<string, unknown>>[]; total: number; paged: boolean; first?: number } | null
  /** The verbosity level; omitted keeps the store's own default. */
  readonly activityLevel?: RefactoryActivityLevel
  readonly activityScroll?: number
  /** The watched run's ledger read, as the face would have written it. */
  readonly steering?: Loadable<RefactorySteering> | null
  /** The watched run's candidates, as the face would have written them. */
  readonly candidates?: Loadable<readonly RefactoryCandidateRow[]> | null
  readonly single?: RefactorySingleState | null
  readonly notice?: RefactoryNotice | null
  readonly busy?: boolean
  /** The tab record's lifetime; a spec aborts it to prove the face stops writing. */
  readonly controller?: AbortController
}

/** What a spec holds after mounting: the rendered view and every hand on the picker. */
export interface Mounted {
  readonly view: RenderResult
  readonly instance: RefactoryStoreInstance
  readonly console: FakeConsole
  readonly face: RefactoryInjected
  readonly controller: AbortController
}

/**
 * Mount the body.
 * @param seed - the tab state to put the store into before rendering.
 * @returns the view, the store instance, the scripted face, and the tab signal.
 */
export function mountBody(seed: TabSeed = {}): Mounted {
  const instance = createRefactoryStore().create()
  const scripted = fakeConsole()
  const face = refactoryFace(scripted.remote)(SESSION, instance.actions)
  const controller = seed.controller ?? new AbortController()
  const actions = instance.actions
  if (seed.bare !== true) actions.start(TAB)
  if (seed.choices !== undefined) {
    if (seed.choices.kind === 'ready') actions.loaded(TAB, seed.choices.value)
    else if (seed.choices.kind === 'loading') actions.loading(TAB)
    else actions.failed(TAB, seed.choices.message)
  }
  if (seed.chosen !== undefined && seed.chosen !== null) actions.chose(TAB, seed.chosen)
  for (const [knob, value] of Object.entries(seed.knobs ?? {})) {
    actions.knob(TAB, knob, value)
  }
  for (const [name, value] of Object.entries(seed.settings ?? {})) {
    actions.setting(TAB, name as keyof typeof NO_SETTINGS, value)
  }
  if (seed.watching !== undefined && seed.watching !== null) actions.watching(TAB, seed.watching)
  if (seed.frame !== undefined && seed.frame !== null) {
    if (seed.frame.kind === 'ready') actions.framed(TAB, seed.frame.value)
    else if (seed.frame.kind === 'failed') actions.frameFailed(TAB, seed.frame.message, seed.frame.code, seed.frame.details)
  }
  // The held history is seeded after `watching`: attaching to a run drops the
  // previous one's history (R-4.4), and what a face writes it writes after that.
  if (seed.activityLevel !== undefined) actions.setActivityLevel(TAB, seed.activityLevel)
  if (seed.activityScroll !== undefined) actions.activityScrolled(TAB, seed.activityScroll)
  if (seed.activity !== undefined && seed.activity !== null) {
    // The run the seed names must be the one the frame names, or the store
    // refuses the page as another run's history (R-4.5).
    const run = seed.frame?.kind === 'ready' ? seed.frame.value.root : seed.watching ?? '/runs/live'
    const { after, cursor, events, total, paged } = seed.activity
    actions.activityPage(TAB, run, { after, cursor, events, total, paged, run: null })
  }
  if (seed.steering !== undefined && seed.steering !== null) {
    if (seed.steering.kind === 'ready') actions.steered(TAB, seed.steering.value)
    else if (seed.steering.kind === 'failed') actions.steeringFailed(TAB, seed.steering.message)
  }
  if (seed.candidates !== undefined && seed.candidates !== null) {
    if (seed.candidates.kind === 'ready') actions.candidatesRead(TAB, seed.candidates.value)
    else if (seed.candidates.kind === 'failed') actions.candidatesFailed(TAB, seed.candidates.message)
    else actions.candidatesLoading(TAB)
  }
  if (seed.single !== undefined && seed.single !== null) {
    if (seed.single.kind === 'ready') actions.ran(TAB, seed.single.value)
    else actions.runLoading(TAB)
  }
  if (seed.notice !== undefined && seed.notice !== null) actions.noticed(TAB, seed.notice)
  if (seed.busy === true) actions.busy(TAB, true)

  const sessions = { byId: { [SESSION]: { cwd: '/work/app' } } } as unknown as SessionListState
  const shared = {
    // A page tab's address is the shell's to mint; the body never reads it.
    useTabInfo: () => ({
      sidebar: { expanded: true, fullscreen: false },
      panel: { id: 'pane-1' },
      tab: {
        id: TAB, kind: 'refactory', contentId: 'refactory', title: zh['type.label'], visible: true,
        navigation: { address: 'refactory', params: undefined, revision: 1 },
        signal: controller.signal,
        actions: { openResource: () => {}, openTab: () => {}, close: () => {} },
      },
    }),
    sessionId: SESSION,
    useSessions: <S,>(sel: (s: SessionListState) => S) => sel(sessions),
    useStore: hookOf(instance),
    actions,
    ...face,
    t: makeTranslate(seed.dictionary ?? zh),
  }
  const view = render(<RefactoryBody {...shared as unknown as RefactoryBodyProps} />)
  return { view, instance, console: scripted, face, controller }
}
