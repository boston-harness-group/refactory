/**
 * The picker's body: one described choice per seam, and the run they start.
 *
 * Everything the picker keeps lives in its store, keyed by tab; everything it
 * asks for goes through its injected face. The body only decides what to draw
 * for each read and what a gesture means: choosing a mode, setting one of the
 * knobs that mode declares, choosing a level-two name, starting the chosen mode,
 * using it once, refreshing or stopping the run being read, and enabling,
 * disabling, removing or importing a package.
 *
 * Nothing here is free text except the two import fields and the knobs a mode
 * declares without a set of accepted values, which the registry and the knob
 * itself require: a mode, a strategy and a package are always chosen from what
 * the Host listed, never typed. A mode the loop cannot search is offered but not
 * started and not used once: the row says which action runs it, and the Host
 * refuses what the loop cannot search.
 */
import { useEffect, useId, useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import type {
  RefactoryCandidateRow,
  RefactoryKnob,
  RefactoryNoteKind,
  RefactoryRunStatus,
  RefactorySteering,
  RefactoryStrategyChoice,
} from '@deepseek-ai/dsh-refactory-console/types'
import type { PropsLocale, PropsRuntime, PropsStore, TranslateNS } from '@deepseek-ai/dsh-client-ui-slots'
import {
  IconPlayOutlineRegular, IconRefreshOutlineRegular, IconStopFillRegular,
} from '@deepseek-ai/dsh-client-ui-primitives'
import type {} from './locales.ts'
import type { RefactoryInjected } from './face.ts'
import type {
  HeldHistory, Loadable, RefactoryActivityLevel, RefactoryBudgetName, RefactoryNotice, RefactorySettingName,
  RefactorySingleState, createRefactoryStore,
} from './store.ts'
import { knobValues, searchBudget, SEARCH_BUDGET_DEFAULTS } from './store.ts'
import css from './RefactoryBody.module.css'
import { GraphControls, RunList } from './GraphControls.tsx'
import { WorkflowArt, stateTone } from './WorkflowArt.tsx'
import { RunActivity } from './RunActivity.tsx'
import { FieldHelp, FieldLabel } from './FieldHelp.tsx'
import { BriefEditor } from './BriefEditor.tsx'
import { SteeringLedger } from './SteeringLedger.tsx'
import { ProfileControls, CandidateComparison, RuntimeControls, DiscoveryControls } from './ProfileControls.tsx'

/** The body's composed props: the tab it draws, its store, its face, and its copy. */
export type RefactoryBodyProps =
  & PropsRuntime<'sidebar.right.pane.tab'>
  & PropsStore<ReturnType<typeof createRefactoryStore>>
  & RefactoryInjected
  & PropsLocale<'sidebarRefactory'>

/**
 * Say why a gesture was refused, in terms of what the reader did.
 *
 * The three refusals the Host names precisely have copy of their own; anything
 * else is the Host's own message, which only the console knows how to phrase.
 * @param t - namespace-bound translate.
 * @param notice - the settled Remote failure.
 * @returns the line to show above the picker.
 */
export function failureLine(t: TranslateNS<'sidebarRefactory'>, notice: RefactoryNotice): string {
  switch (notice.code) {
    case 'refactory/unknown-mode': return t('error.unknownMode')
    case 'refactory/authoring-mode': return t('error.authoringMode')
    case 'refactory/package-disabled': return t('error.packageDisabled')
    case 'refactory/invalid-search-budget': return t('budget.invalid')
    // Carrier and unclassified Host failures reach the reader as themselves:
    // the picker knows nothing useful to add to a console's own message.
    default: return t('error.failed', { message: notice.message })
  }
}

/**
 * What the chosen level-two name says it is, or the default line when none is chosen.
 * @param t - namespace-bound translate.
 * @param chosen - the chosen name, empty for the console's default.
 * @param names - the described choices for this seam.
 * @returns the line under the select.
 */
function descriptionOf(
  t: TranslateNS<'sidebarRefactory'>,
  chosen: string,
  names: readonly RefactoryStrategyChoice[],
): string {
  if (chosen === '') return t('setting.default')
  return names.find(entry => entry.name === chosen)?.description ?? t('setting.default')
}

/** One level-two seam: the named choices, and what the chosen one is. */
function Seam({
  label, id, value, names, t, onChoose,
}: {
  readonly label: string
  readonly id: RefactorySettingName
  readonly value: string
  readonly names: readonly RefactoryStrategyChoice[]
  readonly t: TranslateNS<'sidebarRefactory'>
  readonly onChoose: (value: string) => void
}): ReactNode {
  return (
    <div className={css.field}>
      <FieldLabel htmlFor={`refactory-${id}`} help={t(`help.${id as 'strategy' | 'optimizer' | 'evaluator'}`)}>{label}</FieldLabel>
      <select
        id={`refactory-${id}`}
        className={css.control}
        value={value}
        data-refactory-setting={id}
        onChange={(event) => { onChoose(event.target.value) }}
      >
        <option value="">{t('setting.default')}</option>
        {names.map(entry => <option key={entry.name} value={entry.name}>{entry.name}</option>)}
      </select>
      <p className={css.note} data-refactory-setting-note={id}>{descriptionOf(t, value, names)}</p>
    </div>
  )
}

/** How one declared knob is entered: from a declared set, or as free text of a kind. */
type KnobControl =
  | { readonly kind: 'choice'; readonly values: readonly string[] }
  | { readonly kind: 'number'; readonly step: string; readonly min?: number; readonly max?: number }
  | { readonly kind: 'text' }

/**
 * The control one declared knob needs.
 *
 * A knob that declares values is a choice whatever its kind — an `enum`'s members
 * and a `str` knob's values are both finite sets — and a `bool` is the same kind of
 * choice over the two words the console reads. Everything else is a number, bounded
 * when the declaration gives bounds, or open text.
 * @param knob - the declaration, as the catalogue reports it.
 * @returns which control to draw for it.
 */
function controlOf(knob: RefactoryKnob): KnobControl {
  if (knob.values.length > 0) return { kind: 'choice', values: knob.values }
  if (knob.kind === 'bool') return { kind: 'choice', values: ['true', 'false'] }
  if (knob.kind === 'int' || knob.kind === 'float') {
    const [low, high] = knob.range
    return {
      kind: 'number',
      step: knob.kind === 'int' ? '1' : 'any',
      ...low === null ? {} : { min: low },
      ...high === null ? {} : { max: high },
    }
  }
  return { kind: 'text' }
}

/** One declared knob: its name, the value this run uses, and what it is for. */
function KnobField({
  name, knob, value, t, onChange,
}: {
  readonly name: string
  readonly knob: RefactoryKnob
  readonly value: string
  readonly t: TranslateNS<'sidebarRefactory'>
  readonly onChange: (value: string) => void
}): ReactNode {
  const control = controlOf(knob)
  const unique = useId()
  const id = `refactory-knob-${name}-${unique}`
  return (
    <div className={name === 'brief' ? `${css.field} ${css.fullField}` : css.field}>
      <FieldLabel htmlFor={id} help={knob.description || t('help.knob', { name, kind: knob.kind })}>{name}</FieldLabel>
      {name === 'brief' && control.kind === 'text'
        ? <BriefEditor id={id} value={value} description={knob.description} onChange={onChange} t={t} />
        : control.kind === 'choice'
        ? (
          <select
            id={id}
            className={css.control}
            value={value}
            data-refactory-knob-input={name}
            onChange={(event) => { onChange(event.target.value) }}
          >
            {control.values.map(entry => <option key={entry} value={entry}>{entry}</option>)}
          </select>
        )
        : control.kind === 'number'
          ? (
            <input
              id={id}
              className={css.input}
              type="number"
              step={control.step}
              min={control.min}
              max={control.max}
              value={value}
              data-refactory-knob-input={name}
              onChange={(event) => { onChange(event.target.value) }}
            />
          )
          : (
            <input
              id={id}
              className={css.input}
              type="text"
              value={value}
              data-refactory-knob-input={name}
              onChange={(event) => { onChange(event.target.value) }}
            />
          )}
      {knob.description !== '' && (
        <p className={css.srOnly} data-refactory-knob-description={name}>{knob.description}</p>
      )}
      {knob.frozen && <p className={css.note} data-refactory-knob-frozen={name}>{t('knob.frozen')}</p>}
    </div>
  )
}

/** What using the chosen mode once left, or that the run is still going. */
function Single({
  run, t, onWatch,
}: {
  readonly onWatch: (root: string) => void
  readonly run: RefactorySingleState
  readonly t: TranslateNS<'sidebarRefactory'>
}): ReactNode {
  if (run.kind !== 'ready') {
    const launch = run.launch
    return <div className={css.status} data-refactory-single={run.kind}>
      <p className={css.note}>{t(run.kind === 'failed' ? 'single.failed' : launch?.phase === 'starting' ? 'single.starting' : 'single.running')}</p>
      {launch !== undefined && <>
        <p data-refactory-single-run>{t('single.active', { mode: launch.mode, run: launch.root })}</p>
        <button type="button" className={css.action} disabled={launch.phase !== 'prepared'}
          data-refactory-single-watch onClick={() => { onWatch(launch.root) }}>{t('single.watch')}</button>
        <p className={css.note}>{t('single.watchHelp')}</p>
      </>}
    </div>
  }
  const single = run.value
  const authoring = single.authoring
  return (
    <div className={css.status} data-refactory-single={single.state}>
      <p className={css.statusLine} data-refactory-single-run>{t('single.ran', { mode: single.mode ?? single.run_id, run: single.root })}</p>
      <span className={css.badge} data-tone={stateTone(single.state)}>{t('status.state')}: {single.state}</span>
      <p className={css.statusLine} data-refactory-single-score>
        {single.best_score === null
          ? t(single.finished || ['finished', 'failed', 'died', 'halted'].includes(single.state) ? 'single.noAggregate' : 'status.noScore')
          : t('status.best', { score: single.best_score })}
      </p>
      {single.item_results != null && (
        <p className={css.statusLine} data-refactory-item-results>
          {t('single.itemResults', { ...single.item_results })}
        </p>
      )}
      {authoring !== null && (
        <p className={css.statusLine} data-refactory-single-spine={authoring.spine}>
          {t('single.spine')}: {authoring.spine}
        </p>
      )}
      {authoring !== null && authoring.nodes_due.length > 0 && (
        <p className={css.statusLine} data-refactory-single-due>
          {t('single.due', { nodes: authoring.nodes_due.join(', ') })}
        </p>
      )}
      {authoring !== null && (authoring.spine === 'ready' || authoring.spine === 'refused') && (
        <p className={css.note} data-refactory-single-gate>{t('single.gate')}</p>
      )}
      {authoring !== null && authoring.error !== '' && (
        <p className={css.note} data-refactory-single-error>
          {t('error.failed', { message: authoring.error })}
        </p>
      )}
      {Object.keys(single.facts).length > 0 && (
        <div className={css.status}>
          <p className={css.note}>{t('single.facts')}</p>
          <ul className={css.recorded}>
            {Object.entries(single.facts).map(([name, written]) => (
              <li key={name} data-refactory-fact={name}>{name}: {written}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

/**
 * Why a status could not be read, in the reader's own terms.
 *
 * A payload the collection cap cut and a console that printed something which is
 * not JSON are two different problems — one is a size limit the reader can raise,
 * the other is whatever the console said — so they get two different lines and a
 * refusal never collapses into the other (R-5.4, F-7, F-8).
 * @param t - namespace-bound translate.
 * @param failure - the settled status failure: the Host's code and message.
 * @returns the line to show where the status would be.
 */
export function statusFailureLine(
  t: TranslateNS<'sidebarRefactory'>,
  failure: { readonly code?: string; readonly message: string; readonly details?: unknown },
): string {
  if (failure.code !== 'refactory/console-truncated') return t('error.failed', { message: failure.message })
  const details = failure.details as { capBytes?: unknown; retainedBytes?: unknown } | undefined
  const cap = typeof details?.capBytes === 'number' ? details.capBytes : null
  const retained = typeof details?.retainedBytes === 'number' ? details.retainedBytes : null
  // The seam reports the cap and the retained tail; it never observes a
  // whole-stream size, so the line names only those two facts.
  return cap === null || retained === null
    ? t('error.failed', { message: failure.message })
    : t('activity.truncated', { cap, retained })
}

/** The run's status, once there is a run to read. */
function Status({
  tabId, frame, run, signal, t, refresh, stop, activity, level, scroll,
  setActivityLevel, loadEarlierActivity, scrollActivity,
}: {
  readonly tabId: Parameters<RefactoryInjected['refresh']>[0]
  readonly frame: Loadable<RefactoryRunStatus> | null
  readonly run: string
  readonly signal: AbortSignal
  readonly t: TranslateNS<'sidebarRefactory'>
  readonly refresh: RefactoryInjected['refresh']
  readonly stop: RefactoryInjected['stop']
  readonly activity: HeldHistory | null
  readonly level: RefactoryActivityLevel
  readonly scroll: number
  readonly setActivityLevel: RefactoryInjected['setActivityLevel']
  readonly loadEarlierActivity: RefactoryInjected['loadEarlierActivity']
  readonly scrollActivity: RefactoryInjected['scrollActivity']
}): ReactNode {
  if (frame === null || frame.kind === 'loading') {
    return <p className={css.note} data-refactory-status="reading">{t('status.reading')}</p>
  }
  if (frame.kind === 'failed') {
    return (
      <p className={css.notice} role="alert" data-refactory-status="failed">
        {statusFailureLine(t, frame)}
      </p>
    )
  }
  const status = frame.value
  return (
    <div className={css.status} data-refactory-status={status.state}>
      <span className={css.badge} data-tone={stateTone(status.state)} role="status">{t('status.state')}: {status.state}</span>
      <RunActivity
        activity={status.activity}
        state={status.state}
        level={level}
        held={activity}
        scroll={scroll}
        t={t}
        onLevel={(chosen) => { setActivityLevel(tabId, run, chosen, signal) }}
        onEarlier={(oldest) => { loadEarlierActivity(tabId, run, oldest, signal) }}
        onScroll={(row) => { scrollActivity(tabId, row) }}
      />
      <p className={css.statusLine} data-refactory-score>
        {status.best_score === null
          ? t(status.finished ? 'single.noAggregate' : 'status.noScore')
          : t('status.best', { score: status.best_score })}
      </p>
      {status.authoring != null
        ? <p className={css.statusLine} data-refactory-watched-spine>{t('single.spine')}: {status.authoring.spine}</p>
        : <p className={css.statusLine} data-refactory-evaluations>{t('status.evaluations', { count: status.counters.evaluations })}</p>}
      {status.item_results != null && (
        <p className={css.statusLine} data-refactory-watched-item-results>{t('single.itemResults', { ...status.item_results })}</p>
      )}
      <div className={css.packageActions}>
        <button
          type="button"
          className={css.action}
          data-refactory-refresh
          // The top-level cursor indexes the progress file; the journal continues
          // from where this tab's held history ends, and the two never derive from
          // one another (R-4.2-3, F-3). A run with no journal in its answer has
          // nothing to continue, so it stays the plain status call.
          onClick={() => {
            const through = Math.max(activity?.through ?? 0, status.activity?.cursor ?? 0)
            if (status.activity == null) refresh(tabId, run, status.cursor, signal)
            else refresh(tabId, run, status.cursor, signal, { after: through })
          }}
        >
          <IconRefreshOutlineRegular />
          {t('status.refresh')}
        </button>
        {!status.finished && status.kind === 'optimizer' && (
          <button
            type="button"
            className={css.action}
            data-refactory-stop
            onClick={() => { stop(tabId, run, signal) }}
          >
            <IconStopFillRegular />
            {t('status.stop')}
          </button>
        )}
      </div>
    </div>
  )
}

/** One note kind, and the copy the kind selector shows for it. */
const NOTE_KINDS: readonly RefactoryNoteKind[] = ['info', 'correction', 'constraint']

/**
 * The watched run's candidates, and what a person can do to one.
 *
 * Read on demand rather than on every tick: a leaderboard is a console call of its
 * own, and a search that is not being steered right now does not need its table
 * fetched every few seconds. Every act refreshes it, so the flags a click changes
 * are the flags the person then sees.
 */
function Candidates({
  tabId, run, candidates, finished, signal, t, readCandidates, controlRun,
}: {
  readonly tabId: Parameters<RefactoryInjected['refresh']>[0]
  readonly run: string
  readonly candidates: Loadable<readonly RefactoryCandidateRow[]> | null
  readonly finished: boolean
  readonly signal: AbortSignal
  readonly t: TranslateNS<'sidebarRefactory'>
  readonly readCandidates: RefactoryInjected['readCandidates']
  readonly controlRun: RefactoryInjected['controlRun']
}): ReactNode {
  const [destination, setDestination] = useState('')
  const destinationId = useId()
  return (
    <div className={css.status} data-refactory-candidates={candidates?.kind ?? 'none'}>
      <div className={css.packageActions}>
        <button
          type="button"
          className={css.action}
          data-refactory-candidates-read
          onClick={() => { readCandidates(tabId, run, signal) }}
        >
          <IconRefreshOutlineRegular />
          {t('candidates.read')}
        </button>
        <div className={css.field}>
        <FieldLabel htmlFor={destinationId} help={t('help.destination')}>{t('control.destination')}</FieldLabel>
        <input id={destinationId}
          className={css.input}
          value={destination}
          aria-label={t('control.destination')}
          placeholder={t('control.destination')}
          data-refactory-adopt-name
          onChange={(event) => { setDestination(event.target.value) }}
        />
        </div>
        <button
          type="button"
          className={css.action}
          data-refactory-control="adopt"
          onClick={() => {
            const name = destination.trim()
            controlRun(tabId, run, { control: 'adopt', ...name === '' ? {} : { as: name } }, signal)
          }}
        >
          {t('control.adopt')}
        </button>
      </div>
      {candidates?.kind === 'loading' && (
        <p className={css.note}>{t('candidates.reading')}</p>
      )}
      {candidates?.kind === 'failed' && (
        <p className={css.note}>{t('error.failed', { message: candidates.message })}</p>
      )}
      {candidates?.kind === 'ready' && (candidates.value.length === 0
        ? <p className={css.note} data-refactory-candidates-empty>{t('candidates.empty')}</p>
        : (
          <ul className={css.recorded} data-refactory-candidates-rows>
            {candidates.value.map(row => (
              <li
                key={row.id}
                className={css.package}
                data-refactory-candidate={row.id}
                data-refactory-kept={String(row.kept)}
                data-refactory-rejected={String(row.rejected)}
              >
                <span className={css.name}>{row.id}</span>{' '}
                {row.score === null ? t('candidates.unscored') : String(row.score)}
                {row.kept && <span className={css.note}> · {t('candidate.kept')}</span>}
                {row.rejected && <span className={css.note}> · {t('candidate.rejected')}</span>}
                <div className={css.gestures}>
                  <button
                    type="button"
                    className={css.action}
                    disabled={finished || row.kept}
                    data-refactory-candidate-keep={row.id}
                    onClick={() => {
                      controlRun(tabId, run, { control: 'keep', target: row.id }, signal)
                    }}
                  >
                    {t('candidate.keep')}
                  </button>
                  <button
                    type="button"
                    className={css.action}
                    disabled={finished || row.rejected}
                    data-refactory-candidate-reject={row.id}
                    onClick={() => {
                      controlRun(tabId, run, { control: 'reject', target: row.id }, signal)
                    }}
                  >
                    {t('candidate.reject')}
                  </button>
                </div>
              </li>
            ))}
          </ul>
        ))}
    </div>
  )
}

/** The watched run's intent ledger, and the gates it is stopped at. */
function Steering({
  tabId, run, steering, candidates, optimizer, signal, t, readSteering, readCandidates, addNote,
  answerGate, controlRun, statusFinished,
}: {
  readonly tabId: Parameters<RefactoryInjected['refresh']>[0]
  readonly run: string
  readonly steering: Loadable<RefactorySteering> | null
  readonly candidates: Loadable<readonly RefactoryCandidateRow[]> | null
  readonly optimizer: boolean | null
  readonly signal: AbortSignal
  readonly t: TranslateNS<'sidebarRefactory'>
  readonly readSteering: RefactoryInjected['readSteering']
  readonly readCandidates: RefactoryInjected['readCandidates']
  readonly addNote: RefactoryInjected['addNote']
  readonly answerGate: RefactoryInjected['answerGate']
  readonly controlRun: RefactoryInjected['controlRun']
  readonly statusFinished: boolean | undefined
}): ReactNode {
  const [said, setSaid] = useState('')
  const [kind, setKind] = useState<RefactoryNoteKind>('info')
  const [saving, setSaving] = useState(false)
  const history = <SteeringLedger steering={steering} refresh={() => { readSteering(tabId, run, signal) }} t={t} />
  if (steering === null || steering.kind === 'loading') {
    return <>{history}<p className={css.note} data-refactory-steering="reading">{t('steering.reading')}</p></>
  }
  if (steering.kind === 'failed') {
    return (
      <>{history}<p className={css.note} data-refactory-steering="failed">
        {t('error.failed', { message: steering.message })}
      </p></>
    )
  }
  const ledger = steering.value
  const finished = statusFinished === true || ledger.finished === true
  return (
    <>{history}<div className={css.status} data-refactory-steering="ready">
      {optimizer !== null && ledger.gates.length > 0 && (
        <div className={css.questions} data-refactory-gates="waiting">
          {ledger.gates.map(gate => (
            <div key={gate.id} data-refactory-gate={gate.id}>
              <p className={css.statusLine} data-refactory-gate-prompt>
                {t('steering.gate', { gate: gate.id })}
              </p>
              {gate.prompt !== '' && (
                <p className={`${css.note} ${css.gatePrompt}`} data-refactory-gate-note>{gate.prompt}</p>
              )}
              <div className={css.gestures}>
                {gate.verdicts.map(option => (
                  <button
                    key={option}
                    type="button"
                    className={css.action}
                    data-refactory-verdict={option}
                    onClick={() => { answerGate(tabId, run, gate.id, option, signal) }}
                  >
                    {option}
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
      {optimizer && finished && <p className={css.note} data-refactory-controls-finished>{t('control.finished')}</p>}
      {ledger.control != null && (
        <p className={css.note} data-refactory-control-receipt={ledger.control.status}>
          {t(`control.${ledger.control.status}`, { action: ledger.control.action })}
          {ledger.control.target && <> {ledger.control.target}</>}
          {ledger.control.as && <> · {ledger.control.as}</>}
          {ledger.control.recovered && <> {t('control.recovered')}</>}
          {ledger.control.recovered_control && (
            <> {t('control.recovered')} {ledger.control.recovered_control.action} {ledger.control.recovered_control.target}</>
          )}
          {ledger.control.candidate && <> {ledger.control.candidate}</>}
          {ledger.control.package_root && <> · {ledger.control.package_root}</>}
          {ledger.control.message && <> · {ledger.control.message}</>}
          {ledger.control.error && <> · {ledger.control.error}</>}
        </p>
      )}
      <p className={css.note} data-refactory-steering-run>{ledger.run}</p>
      <div className={css.packageActions}>
        <button
          type="button"
          className={css.action}
          data-refactory-steering-refresh
          onClick={() => { readSteering(tabId, run, signal) }}
        >
          <IconRefreshOutlineRegular />
          {t('steering.refresh')}
        </button>
        {optimizer && <button
          type="button"
          className={css.action}
          data-refactory-control="pause"
          disabled={finished}
          onClick={() => { controlRun(tabId, run, { control: 'pause' }, signal) }}
        >
          {t('control.pause')}
        </button>}
        {optimizer && <button
          type="button"
          className={css.action}
          data-refactory-control="resume"
          disabled={finished}
          onClick={() => { controlRun(tabId, run, { control: 'resume' }, signal) }}
        >
          {t('control.resume')}
        </button>}
        {optimizer && <button
          type="button"
          className={css.action}
          data-refactory-control="stop"
          disabled={finished}
          onClick={() => { controlRun(tabId, run, { control: 'stop' }, signal) }}
        >
          {t('control.stop')}
        </button>}
      </div>
      {optimizer && <Candidates
        key={run}
        tabId={tabId}
        run={run}
        candidates={candidates}
        finished={finished}
        signal={signal}
        t={t}
        readCandidates={readCandidates}
        controlRun={controlRun}
      />}
      {optimizer === false && <p className={css.note}>{t('steering.graphControls')}</p>}
      {optimizer === null && <p className={css.note}>{t('steering.unknownControls')}</p>}
      {ledger.problems.length > 0 && (
        <ul className={css.recorded} data-refactory-ledger="problems">
          {ledger.problems.map((problem, index) => (
            <li key={String(index)} className={css.note} data-refactory-problem>{problem}</li>
          ))}
        </ul>
      )}
      {optimizer !== null && <form
        className={css.import}
        onSubmit={(event) => {
          event.preventDefault()
          const note = said.trim()
          if (note === '' || saving) return
          setSaving(true)
          void addNote(tabId, run, note, kind, signal).then(recorded => {
            if (recorded) setSaid(current => current.trim() === note ? '' : current)
          }).finally(() => { setSaving(false) })
        }}
      >
        <FieldLabel htmlFor={`note-${tabId}`} help={t('help.note')}>{t('steering.note')}</FieldLabel>
        <textarea id={`note-${tabId}`}
          className={`${css.input} ${css.noteDraft}`} rows={3}
          value={said}
          aria-label={t('steering.note')}
          placeholder={t('steering.note')}
          data-refactory-note
          onChange={(event) => { setSaid(event.target.value) }}
        />
        <FieldLabel htmlFor={`note-kind-${tabId}`} help={t('help.noteKind')}>{t('steering.kind')}</FieldLabel>
        <select id={`note-kind-${tabId}`}
          className={css.control}
          value={kind}
          aria-label={t('steering.kind')}
          data-refactory-note-kind
          onChange={(event) => { setKind(event.target.value as RefactoryNoteKind) }}
        >
          {NOTE_KINDS.map(option => (
            <option key={option} value={option}>{t(`kind.${option}`)}</option>
          ))}
        </select>
        <button
          type="submit"
          className={css.action}
          disabled={saving || said.trim() === ''}
          data-refactory-note-add
        >
          {saving ? t('steering.saving') : t('steering.add')}
        </button>
      </form>}

    </div></>
  )
}

/** The picker's body: the described choices, and the run they start. */
export function RefactoryBody({
  useTabInfo, useStore, choose, choosePackage, chooseIntent, setKnob, setSetting, start, run: runOnce, load, reload, refresh, watch, stop,
  readSteering, readCandidates, addNote, answerGate, controlRun, setEnabled, remove,
  importPackage, listRuns, graphControl, profileApi, discoveryApi, t,
  setActivityLevel, loadEarlierActivity, scrollActivity,
}: RefactoryBodyProps): ReactNode {
  const { tab } = useTabInfo()
  const { signal } = tab
  const state = useStore(store => store.byTab[tab.id])
  const [source, setSource] = useState('')
  const [name, setName] = useState('')
  const [watchedPath, setWatchedPath] = useState('')
  const intent = state?.setupIntent ?? 'run'
  const setIntent = (value: 'run' | 'optimize'): void => { chooseIntent(tab.id, value) }
  const setupId = useId()
  useEffect(() => { setWatchedPath(state?.watching ?? '') }, [state?.watching])
  const chosen = state?.choices.kind === 'ready'
    ? state.choices.value.modes.find(mode => mode.id === state.chosen)
    : undefined
  // What both gestures send: every declared knob, with the declaration's own value
  // under anything a person has not changed.
  const knobs = useMemo(() => knobValues(chosen, state?.knobs), [chosen, state?.knobs])
  useEffect(() => {
    // A bucket gone because the record aborted must not be re-seeded by a
    // component that has not unmounted yet.
    if (state !== undefined || signal.aborted) return
    load(tab.id, signal)
  }, [state, tab.id, signal, load])

  if (state === undefined) return null
  const choices = state.choices
  if (choices.kind !== 'ready') {
    return (
      <div className={css.root} data-refactory-state={choices.kind}><div className={css.section}>
        <WorkflowArt className={css.emptyArt} /><h2 className={css.heading}>Refactory</h2>
        <p className={css.statusLine}>
          {choices.kind === 'loading' ? t('choices.loading') : t('error.failed', { message: choices.message })}
        </p>
        {choices.kind === 'failed' && <button className={css.action} onClick={() => { reload(tab.id, signal) }}>{t('reload')}</button>}
      </div></div>
    )
  }
  const { project, modes, strategies, packages, projects } = choices.value
  const selected = chosen
  const packageNames = [...new Set([...packages.map(item => item.name), ...modes.map(mode => mode.package)])].sort()
  const packageModes = modes.filter(mode => mode.package === state.selectedPackage)
  const run = state.watching
  const validBudget = searchBudget(state.settings) !== null
  return (
    <div className={css.root} data-refactory-state="ready">
      <div className={css.header}>
        <WorkflowArt className={css.headerArt} />
        <div className={css.headerCopy}><div className={css.eyebrow}>Refactory</div>
          <h1 className={css.title}>{t('type.label')}</h1>
          <p className={css.note}>{t('panel.tagline')}</p>
        </div>
        <button
          type="button"
          className={css.tool}
          aria-label={t('reload')}
          title={t('reload')}
          data-refactory-reload
          onClick={() => { reload(tab.id, signal) }}
        >
          <IconRefreshOutlineRegular />
        </button>
      </div>
      <details className={css.project} data-refactory-project>
        <summary>{t('project.heading')} · {project.split(/[\/]/).filter(Boolean).at(-1)}</summary>
        <code className={css.projectPath}>{project}</code>
        <p className={css.note}>{t('project.explanation')}</p>
      </details>
      {state.steering?.kind === 'ready' && state.steering.value.gates.length > 0 && <div className={css.attention} role="status">
        <span>{t('panel.questions', { count: state.steering.value.gates.length })}</span>
        <a href={`#refactory-steering-${tab.id}`}>{t('panel.answer')}</a>
      </div>}
      {state.notice !== null && (
        <p className={css.notice} role="alert" data-refactory-notice={state.notice.code}>{failureLine(t, state.notice)}</p>
      )}

      <div className={css.content}>
      <section className={css.section} data-refactory-configuration>
        <h2 className={css.heading}>{t('run.heading')}</h2>
        <div className={css.setupTabs} role="tablist" aria-label={t('setup.label')}>
          {(['run', 'optimize'] as const).map(view => <button key={view} type="button" role="tab"
            id={`${setupId}-${view}-tab`} aria-controls={`${setupId}-${view}-pane`} aria-selected={intent === view}
            tabIndex={intent === view ? 0 : -1} data-refactory-intent={view} onClick={() => { setIntent(view) }}
            onKeyDown={event => {
              if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return
              event.preventDefault()
              const next = event.key === 'Home' ? 'run' : event.key === 'End' ? 'optimize' : view === 'run' ? 'optimize' : 'run'
              setIntent(next)
              document.getElementById(`${setupId}-${next}-tab`)?.focus()
            }}>{t(`setup.${view}`)}</button>)}
        </div>
        <p className={css.note}>{t(intent === 'run' ? 'panel.onceHelp' : 'panel.startHelp')}</p>
        {projects.length > 0 && (!selected || selected.parameters?.target !== undefined) && <div className={css.field}>
          <FieldLabel htmlFor="refactory-target" help={t('help.target')}>{t('target.label')}</FieldLabel>
          <select id="refactory-target" className={css.control} data-refactory-target
            value={state.knobs?.target ?? ''}
            onChange={event => { setKnob(tab.id, 'target', event.target.value) }}>
            <option value="">{t('target.new')}</option>
            {projects.map(p => <option key={p.path} value={p.path}>{p.name} · {p.kind}</option>)}
          </select>
        </div>}
        <div className={css.field}>
          <FieldLabel htmlFor="refactory-package" help={t('help.package')}>{t('package.label')}</FieldLabel>
          <select id="refactory-package" className={css.control} data-refactory-package value={state.selectedPackage ?? ''}
            onChange={event => { choosePackage(tab.id, event.target.value || null) }}>
            <option value="">{t('package.choose')}</option>
            {packageNames.map(name => <option key={name} value={name}>{name} · {modes.filter(mode => mode.package === name).length}</option>)}
          </select>
        </div>
        <div className={css.field}>
          <FieldLabel htmlFor="refactory-mode" help={t('help.mode')}>{t('mode.label')}</FieldLabel>
          <select
            id="refactory-mode"
            className={css.control}
            value={state.chosen ?? ''}
            title={state.chosen ?? ''}
            data-refactory-mode
            disabled={!state.selectedPackage}
            onChange={(event) => { choose(tab.id, event.target.value === '' ? null : event.target.value) }}
          >
            <option value="">{t(state.selectedPackage ? 'mode.choose' : 'package.first')}</option>
            {packageModes.map(mode => (
              <option key={mode.id} value={mode.id}>{mode.mode}</option>
            ))}
          </select>
          {selected !== undefined && (
            <p className={css.note} data-refactory-mode-note={selected.action}>
              {selected.description}
            </p>
          )}
        </div>
        {knobs.length > 0 && <details className={css.disclosure} open data-refactory-inputs>
          <summary>{t('panel.inputs')} · {knobs.length}</summary>
          <div className={css.fields}>
        {[...knobs].sort((a, b) => Number(b.name === 'brief') - Number(a.name === 'brief')).map(({ name: knob, knob: declaration, value }) => (
          <KnobField
            key={`${selected?.id}:${knob}`}
            name={knob}
            knob={declaration}
            value={value}
            t={t}
            onChange={(changed) => { setKnob(tab.id, knob, changed) }}
          />
        ))}
          </div>
        </details>}
        {selected === undefined && <div className={css.empty}><WorkflowArt className={css.emptyArt} /><p>{t('panel.chooseHelp')}</p></div>}
        {selected !== undefined && selected.enabled && selected.action !== 'start' && (
          <p className={css.note} data-refactory-authoring>
            {t('mode.authoring', { action: selected.action })}
          </p>
        )}
        {selected !== undefined && !selected.enabled && (
          <div>
            <p className={css.note} data-refactory-disabled>{t('mode.disabled')}</p>
            <button
              type="button"
              className={css.action}
              disabled={state.busy}
              data-refactory-enable-selected
              onClick={() => { setEnabled(tab.id, selected.package, true, signal) }}
            >
              {t('packages.enable')}
            </button>
          </div>
        )}
        {selected && discoveryApi && <DiscoveryControls api={discoveryApi} mode={selected.id} signal={signal} />}
        {selected && <RuntimeControls configuration={state.settings.configuration || '{}'} onChange={value => { setSetting(tab.id, 'configuration', value) }} />}
        <div role="tabpanel" id={`${setupId}-optimize-pane`} aria-labelledby={`${setupId}-optimize-tab`}
          hidden={intent !== 'optimize'} className={css.setupPane}>
        {selected && !selected.scored && <p className={css.note}>{t('setup.scoredOnly')}</p>}
        {selected?.scored && profileApi && <ProfileControls api={profileApi} mode={selected.id} settings={state.settings} knobs={knobs} knobDefaults={knobValues(selected, {})} signal={signal}
          onField={(key, value) => { setSetting(tab.id, key, value) }} onKnob={(key, value) => { setKnob(tab.id, key, value) }} />}
        <details className={css.disclosure} open data-refactory-search-settings>
          <summary>{t('panel.search')}</summary>
          <div className={css.status}>
        {selected?.action === 'start' && (
          <div className={css.status} data-refactory-search-budget>
            <div className={css.fields}>
            {(Object.keys(SEARCH_BUDGET_DEFAULTS) as RefactoryBudgetName[]).map(name => (
              <div className={css.field} key={name}>
                <FieldLabel htmlFor={`refactory-${name}`} help={t(`help.${name}`)}>{t(`budget.${name}`)}</FieldLabel>
                <input
                  id={`refactory-${name}`}
                  className={css.control}
                  type="number"
                  min={1}
                  max={Number.MAX_SAFE_INTEGER}
                  step={1}
                  value={state.settings[name] ?? SEARCH_BUDGET_DEFAULTS[name]}
                  data-refactory-budget={name}
                  aria-invalid={searchBudget({ [name]: state.settings[name] }) === null}
                  onChange={event => { setSetting(tab.id, name, event.target.value) }}
                />
              </div>
            ))}
            </div>
            <p className={css.note}>{t('budget.scope')}</p>
          </div>
        )}
          <details className={css.disclosure} data-refactory-search-method>
            <summary>{t('setup.method')}</summary>
        <Seam
          id="strategy"
          label={t('strategy.label')}
          value={state.settings.strategy}
          names={strategies.outer_loops}
          t={t}
          onChoose={(value) => { setSetting(tab.id, 'strategy', value) }}
        />
        <Seam
          id="optimizer"
          label={t('optimizer.label')}
          value={state.settings.optimizer}
          names={strategies.optimizers}
          t={t}
          onChoose={(value) => { setSetting(tab.id, 'optimizer', value) }}
        />
        <Seam
          id="evaluator"
          label={t('evaluator.label')}
          value={state.settings.evaluator}
          names={strategies.evaluators}
          t={t}
          onChoose={(value) => { setSetting(tab.id, 'evaluator', value) }}
        />
          </details>
          </div>
        </details>
        {!validBudget && <p className={css.notice} role="alert">{t('budget.invalid')}</p>}
        {selected !== undefined && <div className={css.launchSingle}>
          <p className={css.launchSummary}>{t('setup.optimizeSummary', { mode: selected.id })}</p>
          <button type="button" className={`${css.action} ${css.primary}`}
            disabled={state.starting || state.busy || !selected.enabled || !selected.scored || selected.action !== 'start' || !validBudget}
            data-refactory-start onClick={() => { start(tab.id, selected.id, knobs, state.settings, signal) }}>
            <IconPlayOutlineRegular />{t(state.starting ? 'setup.starting' : 'setup.start')}
          </button>
          <p className={css.note}>{t('budget.scope')}</p>
        </div>}
        </div>
        <div role="tabpanel" id={`${setupId}-run-pane`} aria-labelledby={`${setupId}-run-tab`}
          hidden={intent !== 'run'} className={css.setupPane}>
          {selected?.enabled && <div className={css.launchSingle}>
            <p className={css.launchSummary}>{t('setup.runSummary', { mode: selected.id })}</p>
            <button type="button" className={`${css.action} ${css.primary}`} disabled={state.single?.kind === 'running'}
              data-refactory-run-once onClick={() => { runOnce(tab.id, selected.id, knobs, signal, state.settings) }}>
              <IconPlayOutlineRegular />{t('runOnce')}
            </button>
          </div>}
        </div>
        {state.single !== null && <Single run={state.single} t={t} onWatch={root => { watch(tab.id, root, signal) }} />}
      </section>

      <div className={css.monitor}>
      <section className={css.section}>
        <div className={css.fieldLabel}><h2 className={css.heading}>{t('status.heading')}</h2><FieldHelp label={t('status.heading')}>{t('help.activity')}</FieldHelp></div>
        <form className={css.import} onSubmit={(event) => {
          event.preventDefault()
          watch(tab.id, watchedPath, signal)
        }}>
          <FieldLabel htmlFor={`${setupId}-watch`} help={t('status.watchHelp')}>{t('status.runPath')}</FieldLabel>
          <input
            id={`${setupId}-watch`}
            className={css.input}
            value={watchedPath}
            aria-label={t('status.runPath')}
            placeholder={t('status.runPath')}
            data-refactory-watch-path
            onChange={(event) => { setWatchedPath(event.target.value) }}
          />
          <button type="submit" className={css.action} disabled={watchedPath.trim() === ''} data-refactory-watch>
            {t('status.watch')}
          </button>
        </form>
        {run === null && <p className={css.note}>{t('status.watchHelp')}</p>}
        {listRuns !== undefined && <RunList signal={signal} read={listRuns} watch={root => { watch(tab.id, root, signal) }} />}
        {run !== null && (
          // Which run the gesture produced: the id is what `console status --run`
          // and `wf_state` take, so a person can follow it outside the panel.
          <details className={css.runIdentity}>
            <summary title={run}>{run.split(/[\\/]/).filter(Boolean).at(-1)}</summary>
            <p className={css.note} data-refactory-started>{t('watching', { run })}</p>
            <code className={css.projectPath} data-refactory-run>{run}</code>
          </details>
        )}
        {run === null
          ? <div className={css.empty}><WorkflowArt className={css.emptyArt} /><p className={css.note} data-refactory-status="none">{t('status.none')}</p></div>
          : (
            <Status
              tabId={tab.id}
              frame={state.frame}
              run={run}
              signal={signal}
              t={t}
              refresh={refresh}
              stop={stop}
              activity={state.activity}
              level={state.activityLevel}
              scroll={state.activityScroll}
              setActivityLevel={setActivityLevel}
              loadEarlierActivity={loadEarlierActivity}
              scrollActivity={scrollActivity}
            />
          )}
      </section>

      {run !== null && state.frame?.kind === 'ready' && state.frame.value.kind === 'graph' && graphControl !== undefined &&
        <GraphControls key={run} run={run} signal={signal} control={graphControl} />}
      <section className={css.section} id={`refactory-steering-${tab.id}`}>
        <h2 className={css.heading}>{t('steering.heading')}</h2>
        {run === null
          ? <p className={css.note} data-refactory-steering="none">{t('steering.none')}</p>
          : (
            <Steering
              key={run}
              tabId={tab.id}
              run={run}
              steering={state.steering}
              candidates={state.candidates}
              statusFinished={state.frame?.kind === 'ready' ? state.frame.value.finished : undefined}
              optimizer={state.frame?.kind === 'ready' && state.frame.value.kind !== 'unknown'
                ? state.frame.value.kind === 'optimizer' : null}
              signal={signal}
              t={t}
              readSteering={readSteering}
              readCandidates={readCandidates}
              addNote={addNote}
              answerGate={answerGate}
              controlRun={controlRun}
            />
          )}
      </section>

      {run && profileApi && <div className={css.section}><CandidateComparison key={run} api={profileApi} run={run} signal={signal} /></div>}
      </div>
      </div>
      <details className={`${css.section} ${css.packageSection}`} data-refactory-packages>
        <summary><span>{t('packages.heading')}</span> · {packages.length}</summary>
        <ul className={css.packages}>
          {packages.map(pkg => (
            <li
              key={pkg.name}
              className={css.package}
              data-refactory-package={pkg.name}
              data-refactory-enabled={pkg.enabled === null ? 'unknown' : String(pkg.enabled)}
            >
              <div className={css.packageHead}>
                <span className={css.name}>{pkg.name}</span>
                <span className={css.count}>{t('packages.count', { count: pkg.modes.length })}</span>
              </div>
              <span className={css.state}>
                {pkg.enabled === null
                  ? t('packages.unknown')
                  : pkg.enabled ? t('packages.enabled') : t('packages.disabled')}
              </span>
              <div className={css.packageActions}>
                {pkg.enabled !== null && (
                  <button
                    type="button"
                    className={css.action}
                    disabled={state.busy}
                    data-refactory-toggle={pkg.name}
                    onClick={() => { setEnabled(tab.id, pkg.name, !pkg.enabled, signal) }}
                  >
                    {pkg.enabled ? t('packages.disable') : t('packages.enable')}
                  </button>
                )}
                <button
                  type="button"
                  className={css.action}
                  disabled={state.busy}
                  data-refactory-remove={pkg.name}
                  onClick={() => { remove(tab.id, pkg.name, signal) }}
                >
                  {t('packages.remove')}
                </button>
              </div>
              <ul className={css.packageModes}>
                {pkg.modes.map(mode => <li key={mode.name} data-refactory-package-mode={mode.name}>{mode.name}</li>)}
              </ul>
            </li>
          ))}
        </ul>
        <form
          className={css.import}
          onSubmit={(event) => {
            event.preventDefault()
            importPackage(tab.id, source, name, signal)
            setSource('')
            setName('')
          }}
        >
          <FieldLabel htmlFor={`${setupId}-import-source`} help={t('help.importSource')}>{t('packages.importSource')}</FieldLabel>
          <input id={`${setupId}-import-source`}
            className={css.input}
            value={source}
            aria-label={t('packages.importSource')}
            placeholder={t('packages.importSource')}
            data-refactory-import-source
            onChange={(event) => { setSource(event.target.value) }}
          />
          <FieldLabel htmlFor={`${setupId}-import-name`} help={t('help.importName')}>{t('packages.importName')}</FieldLabel>
          <input id={`${setupId}-import-name`}
            className={css.input}
            value={name}
            aria-label={t('packages.importName')}
            placeholder={t('packages.importName')}
            data-refactory-import-name
            onChange={(event) => { setName(event.target.value) }}
          />
          <button
            type="submit"
            className={css.action}
            disabled={state.busy || source.trim() === ''}
            data-refactory-import
          >
            {t('packages.import')}
          </button>
        </form>
      </details>
    </div>
  )
}
