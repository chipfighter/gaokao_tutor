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
            <h3 className="text-sm font-semibold text-[#3D5A40] mb-4">
              Reasoning Path 可视化
            </h3>
            <ScrollArea className="flex-1">
              <div className="bg-[#F5F3E8] rounded-lg p-6">
                {nodeEvents.length === 0 ? (
                  // Idle state — show placeholder
                  <div className="flex flex-col items-center gap-3">
                    <IdleNode label="等待请求..." />
                    <p className="text-xs text-muted-foreground mt-2">
                      发送消息后，推理路径将实时显示
                    </p>
                  </div>
                ) : (
                  // Dynamic node trail
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
