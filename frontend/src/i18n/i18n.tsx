import { createContext, useContext, useEffect, useMemo, useState } from 'react'

export type Lang = 'en' | 'zh'

type Dictionary = Record<string, string>

const EN: Dictionary = {
  'app.title': 'C2XC-Agent',
  'nav.runs': 'Runs',
  'nav.memories': 'Memories',
  'nav.settings': 'Settings',
  'nav.back': 'Back',

  'runs.newBatch': 'New batch',
  'runs.userRequest': 'User request',
  'runs.nRuns': 'n_runs',
  'runs.recipesPerRun': 'recipes_per_run',
  'runs.temperature': 'temperature',
  'runs.dryRun': 'dry_run',
  'runs.create': 'Create',
  'runs.refresh': 'Refresh',
  'runs.expand': 'Show runs',
  'runs.collapse': 'Hide runs',
  'runs.openRun': 'Open',
  'runs.noBatches': 'No batches yet.',

  'run.status': 'Status',
  'run.createdAt': 'Created',
  'run.startedAt': 'Started',
  'run.endedAt': 'Ended',
  'run.error': 'Error',
  'run.cancel': 'Cancel run',
  'run.rerun': 'Re-run',
  'run.output': 'Output',
  'run.evidence': 'Evidence',
  'run.trace': 'Trace',
  'run.feedback': 'Feedback',
  'run.reasoningbank': 'ReasoningBank',
  'run.outputNotReady': 'Output not ready yet.',
  'run.citations': 'Citations',
  'run.citationsEmpty': 'No citations.',
  'run.openEvidence': 'Open evidence',
  'run.open': 'Open',
  'run.evidenceEmpty': 'No evidence found (no kb_query events yet).',
  'run.evidenceEmptyDryRun': 'This is a dry-run. Evidence is expected to be empty (no KB calls).',
  'run.evidenceEmptyPending': 'No evidence yet. Wait for kb_query events to be recorded.',
  'run.evidenceAfterOutput': 'Evidence is available after output is generated.',
  'run.eventsEmpty': 'No events found.',
  'run.selectAlias': 'Select an alias to view content.',
  'run.selectEvent': 'Select an event to view payload.',
  'run.aliases': 'Aliases',
  'run.events': 'Events',
  'run.failureSummary': 'Failure summary',
  'run.canceledSummary': 'Canceled summary',
  'run.openInTrace': 'Open in Trace',
  'run.traceback': 'Traceback',

  'trace.preset.all': 'All',
  'trace.preset.important': 'Important',
  'trace.preset.llm': 'LLM',
  'trace.preset.kb': 'KB',
  'trace.preset.recap': 'ReCAP',
  'trace.preset.rb': 'RB',

  'batch.cancel': 'Cancel batch',
  'batch.remove': 'Remove batch',

  'common.loading': 'Loading…',
  'common.error': 'Error',
  'common.copy': 'Copy',
  'common.copied': 'Copied',
  'common.loadMore': 'Load more',
  'common.searchPlaceholder': 'Search…',
  'common.noMatches': 'No matches.',
  'common.noopTerminal': 'Already terminal.',
  'common.save': 'Save',
  'common.cancel': 'Cancel',
  'common.edit': 'Edit',
  'common.archive': 'Archive',
  'common.activate': 'Activate',
  'common.new': 'New',
  'common.remove': 'Remove',
  'common.archived': 'archived',
  'common.updatedAt': 'Updated',

  'settings.products': 'Products',
  'settings.presets': 'Presets',

  'products.new': 'New product',
  'products.create': 'Create',
  'products.list': 'Products',
  'products.empty': 'No products yet.',
  'products.namePlaceholder': 'e.g., C2H4, CO, CH4…',

  'presets.new': 'New preset',
  'presets.create': 'Create preset',
  'presets.list': 'Presets',
  'presets.edit': 'Edit preset',
  'presets.empty': 'No presets yet.',
  'presets.name': 'Name',
  'presets.namePlaceholder': 'e.g., C2 products',
  'presets.products': 'Products',
  'presets.selected': 'Selected',
  'presets.count': 'Products',
  'presets.noProducts': 'Create products first (Settings → Products).',

  'feedback.updatedAt': 'Updated',
  'feedback.meta': 'Feedback',
  'feedback.score': 'Score',
  'feedback.scoreHint': 'Optional (e.g., 0–10)',
  'feedback.pros': 'Pros',
  'feedback.cons': 'Cons',
  'feedback.other': 'Other',
  'feedback.products': 'Products',
  'feedback.preset': 'Preset…',
  'feedback.applyPreset': 'Apply',
  'feedback.addProduct': 'Add product',
  'feedback.noProducts': 'No products yet. Create them in Settings → Products.',
  'feedback.noRows': 'No product rows yet.',
  'feedback.product': 'Product',
  'feedback.selectProduct': 'Select…',
  'feedback.value': 'Value',
  'feedback.fraction': 'Fraction',
  'feedback.archivedWarning': 'This product is archived. Keeping it is OK for historical feedback.',
  'feedback.sumZero': 'Total value is 0, so all fractions will be stored as 0.',
  'feedback.save': 'Save feedback',
  'feedback.errorRow': 'Row',
  'feedback.errorMissingProduct': 'Missing product.',
  'feedback.errorDuplicateProduct': 'Duplicate product.',
  'feedback.errorMissingValue': 'Missing or invalid value.',
  'feedback.errorNegative': 'Value must be >= 0.',

  'error.deps.title': 'Missing dependencies for normal run',
  'error.deps.message':
    'The backend refused to queue this run because required KB/LLM configuration is missing.',
  'error.deps.missing': 'Missing',
  'error.deps.fix': 'How to fix',
  'error.deps.fixHint': 'Set the env vars (or install packages) and restart the backend process.',
  'error.deps.dryRunTip': 'Tip: use dry_run=true for UI testing (no KB/LLM calls).',

  'confirm.agentRun.title': 'Start agent run?',
  'confirm.agentRun.message':
    'This action will start a run. If dry_run=false, the backend may call the configured LLM/KB endpoints and may incur API costs.',
  'confirm.removeBatch.title': 'Remove batch from UI?',
  'confirm.removeBatch.message':
    'This will hide the batch and all its runs from the Runs page. Data is kept in SQLite (not deleted).',
  'confirm.confirm': 'Confirm',
  'confirm.cancel': 'Cancel',

  'banner.workerDisabled':
    'Worker is disabled. Runs will stay queued. Start backend with C2XC_ENABLE_WORKER=1.',
  'banner.workerStopped':
    'Worker is not running. Runs may stay queued. Check backend logs or restart the service.',
  'banner.workerApiError':
    'Cannot reach backend worker status. Check if the API is running on :8000 and CORS/proxy settings.',

  'settings.theme': 'Theme',
  'settings.lang': 'Language',
  'settings.light': 'Light',
  'settings.dark': 'Dark',
  'settings.en': 'EN',
  'settings.zh': '中文',

  'memories.list': 'Memories',
  'memories.newManual': 'New manual note',
  'memories.newPlaceholder': 'Write an actionable experience note…',
  'memories.role': 'role',
  'memories.status': 'status',
  'memories.type': 'type',
  'memories.all': 'all',

  'rb.learn': 'Learn from feedback',
  'rb.learnHint': 'Queues an async RB learn job (and strict rollback + re-learn on updates).',
  'rb.openTraceRb': 'Open RB Trace',
  'rb.jobs': 'Jobs',
  'rb.noJobs': 'No RB jobs yet.',
  'rb.jobError': 'Job error',
  'rb.deltas': 'Deltas',
  'rb.rollback': 'Rollback',
  'rb.rollbackLatest': 'Rollback latest',
  'rb.noDeltas': 'No deltas yet.',
}

const ZH: Dictionary = {
  'app.title': 'C2XC-Agent',
  'nav.runs': '运行',
  'nav.memories': '经验库',
  'nav.settings': '设置',
  'nav.back': '返回',

  'runs.newBatch': '新建批次',
  'runs.userRequest': '请求',
  'runs.nRuns': 'n_runs',
  'runs.recipesPerRun': 'recipes_per_run',
  'runs.temperature': 'temperature',
  'runs.dryRun': 'dry_run',
  'runs.create': '创建',
  'runs.refresh': '刷新',
  'runs.expand': '展开 runs',
  'runs.collapse': '收起 runs',
  'runs.openRun': '打开',
  'runs.noBatches': '暂无批次。',

  'run.status': '状态',
  'run.createdAt': '创建时间',
  'run.startedAt': '开始时间',
  'run.endedAt': '结束时间',
  'run.error': '错误',
  'run.cancel': '取消 run',
  'run.rerun': '重跑',
  'run.output': '输出',
  'run.evidence': '证据',
  'run.trace': '追踪',
  'run.feedback': '反馈',
  'run.reasoningbank': '经验库',
  'run.outputNotReady': '输出尚未生成。',
  'run.citations': '引用',
  'run.citationsEmpty': '暂无引用。',
  'run.openEvidence': '打开证据',
  'run.open': '打开',
  'run.evidenceEmpty': '暂无证据（还没有 kb_query 事件）。',
  'run.evidenceEmptyDryRun': '这是 dry-run：不会调用 KB/LLM，因此证据为空是正常的。',
  'run.evidenceEmptyPending': '暂无证据：请等待 kb_query 事件写入。',
  'run.evidenceAfterOutput': '证据会在输出生成后可用。',
  'run.eventsEmpty': '暂无事件。',
  'run.selectAlias': '请选择一个 alias 查看内容。',
  'run.selectEvent': '请选择一个事件查看 payload。',
  'run.aliases': '别名',
  'run.events': '事件',
  'run.failureSummary': '失败摘要',
  'run.canceledSummary': '取消摘要',
  'run.openInTrace': '在 Trace 中打开',
  'run.traceback': '调用堆栈',

  'trace.preset.all': '全部',
  'trace.preset.important': '关键',
  'trace.preset.llm': 'LLM',
  'trace.preset.kb': 'KB',
  'trace.preset.recap': 'ReCAP',
  'trace.preset.rb': 'RB',

  'batch.cancel': '取消 batch',
  'batch.remove': '移除 batch',

  'common.loading': '加载中…',
  'common.error': '错误',
  'common.copy': '复制',
  'common.copied': '已复制',
  'common.loadMore': '加载更多',
  'common.searchPlaceholder': '搜索…',
  'common.noMatches': '无匹配结果。',
  'common.noopTerminal': '已结束，无需取消。',
  'common.save': '保存',
  'common.cancel': '取消',
  'common.edit': '编辑',
  'common.archive': '归档',
  'common.activate': '启用',
  'common.new': '新建',
  'common.remove': '移除',
  'common.archived': '已归档',
  'common.updatedAt': '更新时间',

  'settings.products': '产物目录',
  'settings.presets': '预设',

  'products.new': '新建产物',
  'products.create': '创建',
  'products.list': '产物列表',
  'products.empty': '暂无产物。',
  'products.namePlaceholder': '例如：C2H4、CO、CH4…',

  'presets.new': '新建预设',
  'presets.create': '创建预设',
  'presets.list': '预设列表',
  'presets.edit': '编辑预设',
  'presets.empty': '暂无预设。',
  'presets.name': '名称',
  'presets.namePlaceholder': '例如：C2 产物',
  'presets.products': '产物',
  'presets.selected': '已选',
  'presets.count': '产物数',
  'presets.noProducts': '请先在 设置 → 产物目录 中创建产物。',

  'feedback.updatedAt': '更新时间',
  'feedback.meta': '反馈',
  'feedback.score': '评分',
  'feedback.scoreHint': '可选（例如 0–10）',
  'feedback.pros': '优点',
  'feedback.cons': '缺点',
  'feedback.other': '其他',
  'feedback.products': '产物测量',
  'feedback.preset': '选择预设…',
  'feedback.applyPreset': '应用',
  'feedback.addProduct': '添加产物',
  'feedback.noProducts': '暂无产物：请先到 设置 → 产物目录 创建。',
  'feedback.noRows': '还没有产物行。',
  'feedback.product': '产物',
  'feedback.selectProduct': '请选择…',
  'feedback.value': '数值',
  'feedback.fraction': '比例',
  'feedback.archivedWarning': '该产物已归档：作为历史反馈保留是正常的。',
  'feedback.sumZero': '总和为 0，因此所有 fraction 会被存为 0。',
  'feedback.save': '保存反馈',
  'feedback.errorRow': '行',
  'feedback.errorMissingProduct': '未选择产物。',
  'feedback.errorDuplicateProduct': '产物重复。',
  'feedback.errorMissingValue': '数值缺失或无效。',
  'feedback.errorNegative': '数值必须 >= 0。',

  'error.deps.title': 'Normal run 缺少依赖',
  'error.deps.message': '后端拒绝排队该 run：缺少必要的 KB/LLM 配置。',
  'error.deps.missing': '缺少项',
  'error.deps.fix': '如何修复',
  'error.deps.fixHint': '设置环境变量（或安装依赖）后重启后端进程。',
  'error.deps.dryRunTip': '提示：dry_run=true 可用于 UI 测试（不会调用 KB/LLM）。',

  'confirm.agentRun.title': '确认启动 Agent/LLM 运行？',
  'confirm.agentRun.message':
    '该操作将开始一次 run。若 dry_run=false，后端可能会调用配置的 LLM/KB 接口，并可能产生 API 费用。',
  'confirm.removeBatch.title': '从 UI 中移除该批次？',
  'confirm.removeBatch.message': '该操作会在 Runs 页面隐藏此批次及其 runs（不会真正删除 SQLite 中的数据）。',
  'confirm.confirm': '确认',
  'confirm.cancel': '取消',

  'banner.workerDisabled': 'Worker 未启用：run 会一直处于 queued。请以 C2XC_ENABLE_WORKER=1 启动后端。',
  'banner.workerStopped': 'Worker 未运行：run 可能一直处于 queued。请检查后端日志或重启服务。',
  'banner.workerApiError': '无法获取 worker 状态：请检查后端 API 是否在 :8000 运行，以及代理/CORS 设置。',

  'settings.theme': '主题',
  'settings.lang': '语言',
  'settings.light': '浅色',
  'settings.dark': '深色',
  'settings.en': 'EN',
  'settings.zh': '中文',

  'memories.list': '经验条目',
  'memories.newManual': '新增手工条目',
  'memories.newPlaceholder': '写一条可复用的经验/教训…',
  'memories.role': '角色',
  'memories.status': '状态',
  'memories.type': '类型',
  'memories.all': '全部',

  'rb.learn': '从反馈学习',
  'rb.learnHint': '触发异步 RB 学习（更新反馈会严格回滚并重学）。',
  'rb.openTraceRb': '打开 RB Trace',
  'rb.jobs': '任务',
  'rb.noJobs': '暂无 RB 任务。',
  'rb.jobError': '任务错误',
  'rb.deltas': '变更集',
  'rb.rollback': '回滚',
  'rb.rollbackLatest': '回滚最新',
  'rb.noDeltas': '暂无变更集。',
}

function readInitialLang(): Lang {
  const fromDataset = document.documentElement.dataset.lang
  if (fromDataset === 'en' || fromDataset === 'zh') return fromDataset

  const fromStorage = localStorage.getItem('c2xc_lang')
  if (fromStorage === 'en' || fromStorage === 'zh') return fromStorage

  return 'en'
}

type I18nContextValue = {
  lang: Lang
  setLang: (lang: Lang) => void
  t: (key: string) => string
  toggleLang: () => void
}

const I18nContext = createContext<I18nContextValue | null>(null)

export function I18nProvider(props: { children: React.ReactNode }) {
  const [lang, setLang] = useState<Lang>(() => readInitialLang())

  useEffect(() => {
    document.documentElement.dataset.lang = lang
    document.documentElement.lang = lang
    localStorage.setItem('c2xc_lang', lang)
  }, [lang])

  const dict = lang === 'zh' ? ZH : EN
  const value = useMemo<I18nContextValue>(() => {
    return {
      lang,
      setLang,
      t: (key: string) => dict[key] ?? key,
      toggleLang: () => setLang((v) => (v === 'en' ? 'zh' : 'en')),
    }
  }, [dict, lang])

  return <I18nContext.Provider value={value}>{props.children}</I18nContext.Provider>
}

export function useI18n(): I18nContextValue {
  const ctx = useContext(I18nContext)
  if (!ctx) throw new Error('useI18n must be used within I18nProvider')
  return ctx
}

export function useT(): (key: string) => string {
  return useI18n().t
}
