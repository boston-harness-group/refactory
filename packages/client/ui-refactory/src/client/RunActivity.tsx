/**
 * The watched run's activity: the summary every level shows, the detail the
 * levels above `compact` add, and the journalled history with its paging.
 *
 * Two rules shape the file. The whole history is what the panel shows (DEC-001),
 * so the history disclosure and its paging control are drawn at every level —
 * `compact` hides detail, never the way to the records. And holding is unbounded
 * while *drawing* is not (SD-12): a verbose pass holds every record, but only a
 * `RENDER_WINDOW`-row slice of them is in the DOM, so the tab stays answerable
 * for a run of any size. That is why the list is positioned by a row index the
 * store holds rather than by one `<li>` per record.
 */
import type { ReactNode } from 'react'
import type { RefactoryActivity } from '@deepseek-ai/dsh-refactory-console/types'
import type { TranslateNS } from '@deepseek-ai/dsh-client-ui-slots'
import type {} from './locales.ts'
import css from './RefactoryBody.module.css'
import { RunGraph } from './RunGraph.tsx'
import type { RefactoryActivityLevel, HeldHistory } from './store.ts'
import { ACTIVITY_LEVELS, RENDER_WINDOW } from './store.ts'

const shown = (value: unknown): string => value == null ? 'unavailable' : typeof value === 'string' ? value : JSON.stringify(value)

/**
 * What each level adds to the summary every level draws (§2 R-2's table).
 *
 * `standard` and `verbose` are deliberately the same `true` here: a level decides
 * only how much a row shows, and the two differ in *fetch* behaviour, not in
 * rendering. `verbose` is the level whose gesture exhausts the journal back to
 * its first record (`face.ts` `setActivityLevel` → `exhaust`), while `standard`
 * draws the window a poll has already filled; once both hold the same records
 * they render identically, which is the point — reaching further back must not
 * change what a level looks like, only how far it reaches (R-2.2, R-2.7).
 * A reader looking here for a difference will not find one; it lives in the
 * fetch gesture.
 */
const DETAIL: Readonly<Record<RefactoryActivityLevel, boolean>> = { compact: false, standard: true, verbose: true }

/** One record's row height in pixels, fixed so a row index is a scroll position. */
const ROW_HEIGHT = 24

/** One record's line: the same shape the panel has always drawn, one record per call. */
function recordLine(event: Readonly<Record<string, unknown>>): ReactNode {
  return <>{shown(event.at)} · {shown(event.node_id ?? event.kind)} · {shown(event.state ?? event.status)} {event.summary == null ? '' : shown(event.summary)}</>
}

/**
 * One activity record as a row, addressed by its journal index.
 *
 * A row rather than the list is exported so a spec can assert the bound without
 * a browser frame budget: every drawn record is exactly one `<li>` under
 * `[data-refactory-activity]`, and the renderer draws at most `RENDER_WINDOW` of
 * them (T-18b).
 * @param props - the record, and its index among all held records.
 * @returns the row.
 */
export function ActivityRow({ event, index }: { readonly index: number; readonly event: Readonly<Record<string, unknown>> }): ReactNode {
  return <li data-refactory-activity-row={index}>{recordLine(event)}</li>
}

/** The three-way level control: one visible choice, reachable by keyboard. */
function LevelControl({
  level, onChange, t,
}: {
  readonly level: RefactoryActivityLevel
  readonly onChange: (level: RefactoryActivityLevel) => void
  readonly t: TranslateNS<'sidebarRefactory'>
}): ReactNode {
  return (
    <div className={css.levels} role="radiogroup" aria-label={t('activity.level')}>
      <span className={css.note} aria-hidden="true">{t('activity.level')}</span>
      {ACTIVITY_LEVELS.map(option => (
        <label key={option} className={css.level} data-refactory-activity-level={option}
          data-selected={level === option ? 'true' : 'false'}>
          {/* A radio group is the control assistive technology already knows how
              to use for "exactly one of three", so arrow-key movement and the
              announced position come for free rather than being re-implemented. */}
          <input
            type="radio"
            name="refactory-activity-level"
            value={option}
            checked={level === option}
            onChange={() => { onChange(option) }}
          />
          {t(`activity.${option}`)}
        </label>
      ))}
    </div>
  )
}

/** How the history reads right now: what is held, what is whole, what failed. */
function HistoryState({ held, level, t }: {
  readonly held: HeldHistory
  readonly level: RefactoryActivityLevel
  readonly t: TranslateNS<'sidebarRefactory'>
}): ReactNode {
  return <>
    <p className={css.note} data-refactory-activity-count>
      {t('activity.count', { held: held.events.length, total: held.total })}
      {held.complete ? ` · ${t('activity.complete')}` : ''}
    </p>
    {!held.paged && <p className={css.note} data-refactory-activity-unpaged>{t('activity.unpagedNote')}</p>}
    {held.reading && <p className={css.note} data-refactory-activity-reading>{t('activity.reading')}</p>}
    {held.error !== null && <p className={css.notice} role="alert" data-refactory-activity-error>
      {t('error.failed', { message: held.error })}
    </p>}
    {level === 'verbose' && !held.complete && held.paged && !held.reading && held.error === null && (
      <p className={css.note} data-refactory-activity-partial>{t('activity.partial')}</p>
    )}
    {level !== 'verbose' && !held.complete && held.paged && <p className={css.note} data-refactory-activity-more>
      {t('activity.more')}
    </p>}
  </>
}

/**
 * The windowed list over the held records.
 *
 * The scroll container is as tall as the whole history would be, and only the
 * slice at the scroll position is drawn: `RENDER_WINDOW` rows at a time, with the
 * drawn slice translated by the rows above it so the scrollbar keeps its real
 * length. Nothing here is proportional to `N` except that height.
 */
function HistoryWindow({
  held, level, scroll, onScroll, t,
}: {
  readonly held: HeldHistory
  readonly level: RefactoryActivityLevel
  readonly scroll: number
  readonly onScroll: (row: number) => void
  readonly t: TranslateNS<'sidebarRefactory'>
}): ReactNode {
  const events = held.events
  // `standard` opens at the newest records (SD-2); once the reader scrolls,
  // both levels follow the scroll position (SD-12), so the drawn slice is always
  // where the reader is looking rather than pinned to one end of the history.
  const last = Math.max(0, events.length - RENDER_WINDOW)
  const requested = level === 'verbose' ? scroll : Math.max(scroll, last)
  const first = Math.min(Math.max(0, requested), last)
  // The spacer puts the drawn slice at its position in the whole history, so the
  // scrollbar spans the run. The viewport is scrolled to that same position:
  // translating without scrolling left a fresh view scrolled to the top (0) with
  // its rows thousands of pixels down, which read as a large empty gap above the
  // first history row.
  const offset = first * ROW_HEIGHT
  return (
    <div
      className={css.historyViewport}
      // The scrollable height is the whole history's, so the scrollbar is the
      // history and the drawn rows are only the slice at the reader's position.
      style={{ height: Math.max(1, events.length) * ROW_HEIGHT }}
      tabIndex={0}
      role="group"
      aria-label={t('activity.history')}
      data-refactory-activity-window={first}
      // The offset the viewport is kept at. Exposed so a test can assert that the
      // scroll position and the drawn slice agree, which is what the reported
      // empty gap above the first row violated.
      data-refactory-activity-offset={offset}
      // Opening the history and switching levels move the drawn slice without a
      // scroll event, so the viewport is placed at it then. A page arrival must
      // NOT reposition: the reader may be reading further back, and the key
      // therefore carries the level only.
      key={level}
      ref={(node) => { if (node !== null && node.scrollTop !== offset) node.scrollTop = offset }}
      onScroll={(event) => {
        // Only the row index travels, and only when the reader has moved a whole
        // window of rows: a render per scrolled pixel would be work proportional
        // to the input rather than to the window (R-2.5).
        const row = Math.floor(event.currentTarget.scrollTop / (ROW_HEIGHT * RENDER_WINDOW)) * RENDER_WINDOW
        if (row !== first) onScroll(row)
      }}
    >
      <ol className={css.recorded} style={{ transform: `translateY(${offset}px)` }}>
        {events.slice(first, first + RENDER_WINDOW).map((event, index) => (
          <ActivityRow key={first + index} event={event} index={first + index} />
        ))}
      </ol>
    </div>
  )
}

/** The held history as the renderer sees it, falling back to the answer's own page. */
function historyOf(activity: RefactoryActivity, held: HeldHistory | null): HeldHistory {
  if (held !== null && held.run === activity.run_id) return held
  const paged = activity.paged !== false && typeof activity.has_more === 'boolean'
  return {
    run: activity.run_id,
    events: activity.events,
    first: typeof activity.after === 'number' ? activity.after : 0,
    through: activity.cursor,
    total: activity.total ?? activity.events.length,
    complete: !paged || activity.has_more !== true,
    paged,
    reading: false,
    error: null,
  }
}

/**
 * The activity section for one watched run.
 * @param props - the run's activity, the held history, the level, and the gestures.
 * @returns the section, or nothing when the run answered with no activity.
 */
export function RunActivity({
  activity, state = '', level = 'standard', held = null, scroll = 0,
  onLevel = () => {}, onScroll = () => {}, onEarlier = () => {},
  t,
}: {
  readonly state?: string
  readonly activity: RefactoryActivity | null | undefined
  readonly level?: RefactoryActivityLevel
  readonly held?: HeldHistory | null
  readonly scroll?: number
  readonly onLevel?: (level: RefactoryActivityLevel) => void
  readonly onScroll?: (row: number) => void
  readonly onEarlier?: (oldest: number) => void
  readonly t: TranslateNS<'sidebarRefactory'>
}): ReactNode {
  if (activity == null) return null
  const detail = DETAIL[level]
  const history = historyOf(activity, held)
  // "Load earlier activity" has nothing left to fetch once the held history
  // reaches journal index 0 — that is the test, not "as many records as `total`
  // says", because a held history may begin past 0 with every later record held.
  const atStart = history.first === 0 || !history.paged
  return <section className={css.status} aria-label="Live graph activity" data-refactory-activity data-refactory-level={level}>
    <div className={css.activityHead}>
      <h3 className={css.heading}>Live graph activity</h3>
      <LevelControl level={level} onChange={onLevel} t={t} />
    </div>
    <RunGraph activity={activity} state={state} />
    {!activity.graph && <ul className={css.nodeTrack} aria-label="Recorded nodes">
      {activity.completed.map(node => <li className={css.node} data-state="complete" key={'done-' + node}><span aria-hidden="true">✓</span>{node}<span className={css.note}>complete</span></li>)}
      {activity.nodes.filter(node => !activity.completed.includes(node)).map(node => <li className={css.node} data-state="current" key={node}><span aria-hidden="true">◇</span>{node}<span className={css.note}>current</span></li>)}
    </ul>}
    <dl className={css.activityMeta}>
      <dt>Last event</dt><dd>{activity.last_event_at ?? 'no event recorded'}</dd>
      <dt>Waiting</dt><dd>{activity.wait_reason ?? 'no wait recorded'}</dd>
      <dt>Current nodes</dt><dd>{activity.nodes.join(', ') || 'none'}</dd>
      <dt>Completed</dt><dd>{activity.completed.length ? activity.completed.length + ' recorded' : 'none recorded'}</dd>
    </dl>
    <p className={css.note}>Recorded activity, without an estimated percentage. Opening this view only reads the saved run.</p>
    {detail && (activity.resources == null ? <p className={css.note}>Resource usage: unavailable</p>
      : <details className={css.disclosure}><summary>Resource usage and limits</summary><pre>{JSON.stringify(activity.resources, null, 2)}</pre></details>)}
    {/* The worker line is the one-line route/runtime summary every level keeps;
        the usage and full-receipt disclosures are the detail `compact` folds
        away (§2 R-2's "per-worker route/runtime summary" row). */}
    {activity.workers.map((worker, index) => <article className={css.package} key={String(worker.worker_id ?? worker.id ?? index)} data-refactory-worker>
      <p className={css.name}>{shown(worker.node_id || worker.node || worker.role)} · {shown(worker.state ?? worker.status)}</p>
      {detail && <>
        <p className={css.note}>Actual route: {shown(worker.actual_route)} · Runtime: {shown(worker.runtime ?? worker.version)}</p>
        <p className={css.note}>Usage: {shown(worker.usage)}</p>
        <details><summary>Worker capabilities and requested route</summary><pre>{JSON.stringify(worker, null, 2)}</pre></details>
      </>}
    </article>)}
    {detail && activity.deliveries.length > 0 && <details><summary>Instruction delivery</summary>
      <p className={css.note}>A supplied context is not proof of dispatch or of output compliance.</p>
      <ul className={css.recorded}>{activity.deliveries.map((delivery, index) => <li key={String(delivery.worker_id ?? index)}>
        {shown(delivery.node_id)} · {shown(delivery.state)} · revision {shown(delivery.context_revision)}
      </li>)}</ul>
    </details>}
    {detail && activity.control != null && <details><summary>Control receipt · {shown(activity.control.status)}</summary><pre>{JSON.stringify(activity.control, null, 2)}</pre></details>}
    {/* Kept at every level above compact and never truncated: E-1's Option A
        leaves `outputs` in the answer, so its disclosure stays (T-23b). */}
    {detail && <details data-refactory-activity-outputs><summary>Outputs and validation</summary><pre>{JSON.stringify(activity.outputs, null, 2)}</pre></details>}
    <details className={css.disclosure} data-refactory-activity-history>
      <summary>{t('activity.history')} · {history.events.length}/{history.total}</summary>
      <HistoryState held={history} level={level} t={t} />
      {/* The paging control is drawn at every level, `compact` included: the
          history is part of what the panel shows, so it never becomes
          unreachable without first changing level (R-2.4). */}
      <div className={css.gestures}>
        <button type="button" className={css.action} data-refactory-activity-earlier
          disabled={history.reading || atStart}
          onClick={() => { onEarlier(history.first) }}>
          {t('activity.earlier')}
        </button>
      </div>
      {history.events.length > 0 && <HistoryWindow held={history} level={level} scroll={scroll} onScroll={onScroll} t={t} />}
    </details>
  </section>
}
