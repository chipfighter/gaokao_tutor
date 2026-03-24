"use client"

import { useState, useEffect, useRef } from "react"
import { ChevronLeft, ChevronRight } from "lucide-react"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { cn } from "@/lib/utils"

// ── Exported types consumed by page.tsx ────────────────────────────

export interface LogEntry {
  type: "info" | "error" | "warning" | "perf" | "usage"
  message: string
  ts: string
}

export interface NodeEvent {
  node: string
  status: "running" | "done"
  ts: string
  endTs?: string
  durationMs?: number
}

interface RightPanelProps {
  logs: LogEntry[]
  nodeEvents: NodeEvent[]
  tokenUsage: { input: number; output: number; total: number }
}

// ── Human-readable node labels ─────────────────────────────────────

const NODE_LABELS: Record<string, string> = {
  supervisor: "意图分类",
  academic_router: "学术路由",
  rag_retrieve: "RAG 检索",
  web_search: "网络搜索",
  generate_answer: "回答生成",
  evaluate_hallucination: "幻觉评估",
  search_policy: "政策搜索",
  generate_plan: "计划生成",
  emotional_response: "情绪支持",
}

// ── Main component ─────────────────────────────────────────────────

export function RightPanel({ logs, nodeEvents, tokenUsage }: RightPanelProps) {
  const [isCollapsed, setIsCollapsed] = useState(true)
  const [viewTab, setViewTab] = useState<"trail" | "graph">("trail")
  const logsEndRef = useRef<HTMLDivElement>(null)

  // Auto-scroll logs to bottom when new entries arrive
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [logs])

  return (
    <div
      className={cn(
        "relative h-full border-l border-border bg-sidebar flex flex-col",
        "transition-all duration-300 ease-in-out",
        isCollapsed ? "w-12" : "w-80"
      )}
    >
      {isCollapsed ? (
        <Button
          variant="ghost"
          size="icon"
          onClick={() => setIsCollapsed(false)}
          className="absolute top-4 left-1 h-8 w-8 text-muted-foreground hover:text-foreground"
        >
          <ChevronLeft className="h-4 w-4" />
        </Button>
      ) : (
        <>
          {/* Collapse Button */}
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setIsCollapsed(true)}
            className="absolute top-4 left-2 h-8 w-8 text-muted-foreground hover:text-foreground z-10"
          >
            <ChevronRight className="h-4 w-4" />
          </Button>

          {/* Reasoning Path Visualization - 70% height */}
          <div className="p-4 pl-12 flex-[7] flex flex-col border-b border-border">
            {/* Tab toggle */}
            <div className="flex items-center gap-2 mb-3">
              <button
                onClick={() => setViewTab("trail")}
                className={cn(
                  "text-xs px-2 py-1 rounded transition-colors",
                  viewTab === "trail"
                    ? "bg-[#3D5A40] text-white"
                    : "text-[#3D5A40] hover:bg-[#3D5A40]/10"
                )}
              >
                Node Trail
              </button>
              <button
                onClick={() => setViewTab("graph")}
                className={cn(
                  "text-xs px-2 py-1 rounded transition-colors",
                  viewTab === "graph"
                    ? "bg-[#3D5A40] text-white"
                    : "text-[#3D5A40] hover:bg-[#3D5A40]/10"
                )}
              >
                Graph View
              </button>
            </div>

            <ScrollArea className="flex-1">
              {viewTab === "trail" ? (
                <div className="bg-[#F5F3E8] rounded-lg p-6">
                  {nodeEvents.length === 0 ? (
                    <div className="flex flex-col items-center gap-3">
                      <IdleNode label="等待请求..." />
                      <p className="text-xs text-muted-foreground mt-2">
                        发送消息后，推理路径将实时显示
                      </p>
                    </div>
                  ) : (
                    <div className="flex flex-col items-center gap-1">
                      {nodeEvents.map((event, idx) => (
                        <div key={`${event.node}-${idx}`} className="flex flex-col items-center">
                          {idx > 0 && <ArrowDown />}
                          <TraversalNode event={event} />
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ) : (
                <div className="bg-[#F5F3E8] rounded-lg p-3">
                  <GraphDAGView nodeEvents={nodeEvents} />
                </div>
              )}
            </ScrollArea>
          </div>

          {/* Token Usage Counter */}
          {tokenUsage.total > 0 && (
            <div className="px-4 py-2 pl-12 border-b border-border bg-[#F5F3E8]/50">
              <p className="text-xs font-mono text-[#3D5A40]">
                Tokens: {tokenUsage.total}
                <span className="text-muted-foreground ml-1">
                  (in: {tokenUsage.input} / out: {tokenUsage.output})
                </span>
              </p>
            </div>
          )}

          {/* System Logs - 30% height */}
          <div className="flex-[3] flex flex-col overflow-hidden min-h-0">
            <div className="px-4 py-3">
              <h3 className="text-sm font-semibold text-[#3D5A40]">系统 Logs</h3>
            </div>
            <ScrollArea className="flex-1 px-4">
              <div className="flex flex-col gap-1 pb-4">
                {logs.map((log, index) => (
                  <div
                    key={index}
                    className={cn(
                      "text-xs font-mono py-1 px-2 rounded flex gap-2",
                      log.type === "error" && "text-[#D97B6C] bg-[#D97B6C]/10",
                      log.type === "info" && "text-muted-foreground bg-[#F5F3E8]",
                      log.type === "warning" && "text-[#B8860B] bg-[#FFCC99]/20",
                      log.type === "perf" && "text-[#4A90D9] bg-[#4A90D9]/10",
                      log.type === "usage" && "text-[#8B5CF6] bg-[#8B5CF6]/10"
                    )}
                  >
                    <span className="opacity-50 shrink-0">{log.ts}</span>
                    <span>{log.message}</span>
                  </div>
                ))}
                <div ref={logsEndRef} />
              </div>
            </ScrollArea>
          </div>
        </>
      )}
    </div>
  )
}

// ── Graph DAG View ────────────────────────────────────────────────

const DAG_W = 252
const DAG_H = 272
const N_W = 56
const N_H = 26

interface DagNodeDef {
  id: string
  cx: number
  cy: number
}

const DAG_NODE_DEFS: DagNodeDef[] = [
  { id: "supervisor", cx: 126, cy: 16 },
  { id: "academic_router", cx: 56, cy: 64 },
  { id: "search_policy", cx: 156, cy: 64 },
  { id: "emotional_response", cx: 222, cy: 64 },
  { id: "rag_retrieve", cx: 28, cy: 116 },
  { id: "web_search", cx: 84, cy: 116 },
  { id: "generate_plan", cx: 156, cy: 116 },
  { id: "generate_answer", cx: 56, cy: 168 },
  { id: "evaluate_hallucination", cx: 56, cy: 220 },
]

interface DagEdgeDef {
  from: string
  to: string
  retry?: boolean
}

const DAG_EDGE_DEFS: DagEdgeDef[] = [
  { from: "supervisor", to: "academic_router" },
  { from: "supervisor", to: "search_policy" },
  { from: "supervisor", to: "emotional_response" },
  { from: "academic_router", to: "rag_retrieve" },
  { from: "academic_router", to: "web_search" },
  { from: "rag_retrieve", to: "generate_answer" },
  { from: "web_search", to: "generate_answer" },
  { from: "generate_answer", to: "evaluate_hallucination" },
  { from: "search_policy", to: "generate_plan" },
  { from: "evaluate_hallucination", to: "academic_router", retry: true },
]

const nodePos = (id: string) => DAG_NODE_DEFS.find((n) => n.id === id)!

function GraphDAGView({ nodeEvents }: { nodeEvents: NodeEvent[] }) {
  // Derive node states from the latest event per node
  const nodeStates = new Map<
    string,
    { state: "idle" | "running" | "done"; durationMs?: number }
  >()
  for (const def of DAG_NODE_DEFS) {
    let found: NodeEvent | undefined
    for (let i = nodeEvents.length - 1; i >= 0; i--) {
      if (nodeEvents[i].node === def.id) {
        found = nodeEvents[i]
        break
      }
    }
    if (!found) nodeStates.set(def.id, { state: "idle" })
    else if (found.status === "running") nodeStates.set(def.id, { state: "running" })
    else nodeStates.set(def.id, { state: "done", durationMs: found.durationMs })
  }

  // Retry is active when academic_router appears more than once
  const retryActive = nodeEvents.filter((e) => e.node === "academic_router").length > 1

  const edgeLine = (from: DagNodeDef, to: DagNodeDef, active: boolean) => (
    <line
      x1={from.cx}
      y1={from.cy + N_H / 2}
      x2={to.cx}
      y2={to.cy - N_H / 2}
      stroke={active ? "#3D5A40" : "#7A9E7E"}
      strokeWidth={active ? 1.5 : 1}
      strokeDasharray={active ? "none" : "4 2"}
      markerEnd={active ? "url(#dag-arrow-active)" : "url(#dag-arrow)"}
      opacity={active ? 1 : 0.4}
    />
  )

  const endEdge = (x1: number, y1: number, x2: number, y2: number, active: boolean) => (
    <line
      x1={x1}
      y1={y1}
      x2={x2}
      y2={y2}
      stroke={active ? "#3D5A40" : "#7A9E7E"}
      strokeWidth={1}
      strokeDasharray={active ? "none" : "4 2"}
      opacity={active ? 0.7 : 0.3}
    />
  )

  return (
    <div className="relative" style={{ width: DAG_W, height: DAG_H }}>
      {/* SVG edge layer */}
      <svg
        className="absolute inset-0 pointer-events-none"
        width={DAG_W}
        height={DAG_H}
      >
        <defs>
          <marker
            id="dag-arrow"
            viewBox="0 0 6 6"
            refX="6"
            refY="3"
            markerWidth="5"
            markerHeight="5"
            orient="auto"
          >
            <path d="M0,0 L6,3 L0,6 Z" fill="#7A9E7E" />
          </marker>
          <marker
            id="dag-arrow-active"
            viewBox="0 0 6 6"
            refX="6"
            refY="3"
            markerWidth="5"
            markerHeight="5"
            orient="auto"
          >
            <path d="M0,0 L6,3 L0,6 Z" fill="#3D5A40" />
          </marker>
        </defs>

        {/* Normal edges */}
        {DAG_EDGE_DEFS.filter((e) => !e.retry).map((edge) => {
          const from = nodePos(edge.from)
          const to = nodePos(edge.to)
          const targetState = nodeStates.get(edge.to)?.state
          const active = targetState === "running" || targetState === "done"
          return (
            <g key={`${edge.from}-${edge.to}`}>
              {edgeLine(from, to, active)}
            </g>
          )
        })}

        {/* Retry edge — dashed curve from eval left side back up to academic left side */}
        <path
          d={`M ${56 - N_W / 2} ${220} C 4 170, 4 115, ${56 - N_W / 2} ${64}`}
          fill="none"
          stroke={retryActive ? "#D97B6C" : "#7A9E7E"}
          strokeWidth={retryActive ? 1.5 : 1}
          strokeDasharray="4 2"
          markerEnd="url(#dag-arrow)"
          opacity={retryActive ? 0.8 : 0.3}
        />

        {/* END edges */}
        {endEdge(222, 64 + N_H / 2, 222, 108, nodeStates.get("emotional_response")?.state === "done")}
        {endEdge(156, 116 + N_H / 2, 156, 160, nodeStates.get("generate_plan")?.state === "done")}
        {endEdge(56 + N_W / 2, 220, 126, 252, nodeStates.get("evaluate_hallucination")?.state === "done")}
      </svg>

      {/* Node layer */}
      {DAG_NODE_DEFS.map((def) => {
        const { state, durationMs } = nodeStates.get(def.id)!
        const label = NODE_LABELS[def.id] || def.id
        return (
          <div
            key={def.id}
            className={cn(
              "absolute rounded border text-center flex flex-col items-center justify-center",
              "transition-all duration-300",
              state === "idle" &&
                "border-dashed border-[#7A9E7E]/50 bg-white/50 text-muted-foreground",
              state === "running" &&
                "border-[#E8A87C] bg-[#FFCC99] text-[#5C3D2E] font-semibold animate-pulse",
              state === "done" &&
                "border-[#3D5A40] bg-[#3D5A40]/10 text-[#3D5A40]"
            )}
            style={{
              left: def.cx - N_W / 2,
              top: def.cy - N_H / 2,
              width: N_W,
              height: N_H,
            }}
          >
            <span className="text-[8px] leading-tight">{label}</span>
            {state === "done" && durationMs != null && (
              <span className="text-[7px] opacity-60 leading-none">{durationMs}ms</span>
            )}
          </div>
        )
      })}

      {/* END markers */}
      <span
        className="absolute text-[8px] font-bold text-[#7A9E7E]/60"
        style={{ left: 213, top: 110 }}
      >
        END
      </span>
      <span
        className="absolute text-[8px] font-bold text-[#7A9E7E]/60"
        style={{ left: 147, top: 162 }}
      >
        END
      </span>
      <span
        className="absolute text-[8px] font-bold text-[#7A9E7E]/60"
        style={{ left: 118, top: 252 }}
      >
        END
      </span>

      {/* Retry label */}
      <span
        className="absolute text-[7px] italic text-[#D97B6C]/60"
        style={{ left: 0, top: 138 }}
      >
        retry
      </span>
    </div>
  )
}

// ── Sub-components ─────────────────────────────────────────────────

function TraversalNode({ event }: { event: NodeEvent }) {
  const label = NODE_LABELS[event.node] || event.node
  const isRunning = event.status === "running"

  return (
    <div
      className={cn(
        "px-4 py-2 rounded-lg border-2 text-xs font-medium w-40 text-center",
        "transition-all duration-300",
        isRunning
          ? "bg-[#FFCC99] border-[#E8A87C] text-[#5C3D2E] font-semibold animate-pulse"
          : "bg-[#3D5A40]/10 border-[#3D5A40] text-[#3D5A40]"
      )}
    >
      <div className="flex items-center justify-center gap-1.5">
        {isRunning ? (
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#E8A87C] opacity-75" />
            <span className="relative inline-flex rounded-full h-2 w-2 bg-[#E8A87C]" />
          </span>
        ) : (
          <svg className="h-3 w-3 text-[#3D5A40]" viewBox="0 0 12 12" fill="none">
            <path d="M2 6l3 3 5-5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        )}
        {label}
      </div>
      <div className="text-[10px] opacity-60 mt-0.5">
        {isRunning
          ? event.ts
          : `${event.ts} → ${event.endTs ?? ""}${event.durationMs != null ? ` (${event.durationMs}ms)` : ""}`}
      </div>
    </div>
  )
}

function IdleNode({ label }: { label: string }) {
  return (
    <div className="px-4 py-2 rounded-lg border-2 border-dashed border-[#7A9E7E]/50 text-xs font-medium text-muted-foreground">
      {label}
    </div>
  )
}

function ArrowDown() {
  return (
    <div className="flex flex-col items-center text-[#7A9E7E] my-0.5">
      <div className="w-0.5 h-3 bg-[#7A9E7E]/50" />
      <svg width="8" height="6" viewBox="0 0 8 6" fill="currentColor" className="opacity-70">
        <path d="M4 6L0 0h8L4 6z" />
      </svg>
    </div>
  )
}
