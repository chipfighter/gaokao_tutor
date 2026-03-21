"use client"

import { useState } from "react"
import { ChevronLeft, ChevronRight } from "lucide-react"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { cn } from "@/lib/utils"

interface LogEntry {
  type: "info" | "error" | "warning"
  message: string
}

interface RightPanelProps {
  logs: LogEntry[]
}

export function RightPanel({ logs }: RightPanelProps) {
  const [isCollapsed, setIsCollapsed] = useState(true)

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
          {/* Collapse Button - Top Left */}
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
            <h3 className="text-sm font-semibold text-[#3D5A40] mb-4">Reasoning Path 可视化</h3>
            <div className="bg-[#F5F3E8] rounded-lg p-6 flex-1 flex items-center justify-center">
              <div className="flex flex-col items-center gap-3">
                {/* Graph Visualization with new colors */}
                <GraphNode label="用户输入" color="sage" />
                <ArrowDown />
                <GraphNode label="检索器" color="coral" />
                <ArrowDown />
                <GraphNode label="LLM生成" color="coralHighlight" />
                <ArrowDown />
                <GraphNode label="最终输出" color="sageLight" />
              </div>
            </div>
          </div>

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
                      "text-xs font-mono py-1 px-2 rounded",
                      log.type === "error" && "text-[#D97B6C] bg-[#D97B6C]/10",
                      log.type === "info" && "text-muted-foreground bg-[#F5F3E8]",
                      log.type === "warning" && "text-[#B8860B] bg-[#FFCC99]/20"
                    )}
                  >
                    {log.message}
                  </div>
                ))}
              </div>
            </ScrollArea>
          </div>
        </>
      )}
    </div>
  )
}

function GraphNode({ label, color }: { label: string; color: "sage" | "coral" | "coralHighlight" | "sageLight" }) {
  const colorClasses = {
    sage: "bg-[#3D5A40]/10 border-[#3D5A40] text-[#3D5A40]",
    coral: "bg-[#FFCC99]/30 border-[#E8A87C] text-[#8B5A3C]",
    coralHighlight: "bg-[#FFCC99] border-[#E8A87C] text-[#5C3D2E] font-semibold",
    sageLight: "bg-[#7A9E7E]/20 border-[#7A9E7E] text-[#3D5A40]"
  }

  return (
    <div className={cn(
      "px-4 py-2 rounded-lg border-2 text-xs font-medium",
      colorClasses[color]
    )}>
      {label}
    </div>
  )
}

function ArrowDown() {
  return (
    <div className="flex flex-col items-center text-[#7A9E7E]">
      <div className="w-0.5 h-3 bg-[#7A9E7E]/50" />
      <svg width="8" height="6" viewBox="0 0 8 6" fill="currentColor" className="opacity-70">
        <path d="M4 6L0 0h8L4 6z" />
      </svg>
    </div>
  )
}
