"use client"

import { useState } from "react"
import { ChevronLeft, ChevronRight, MessageSquarePlus, MessageSquare, Settings } from "lucide-react"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"

interface ChatHistoryItem {
  id: string
  title: string
}

interface LeftSidebarProps {
  chatHistory: ChatHistoryItem[]
  onNewChat: () => void
  onSelectChat: (id: string) => void
  selectedChatId?: string
}

export function LeftSidebar({ chatHistory, onNewChat, onSelectChat, selectedChatId }: LeftSidebarProps) {
  const [isCollapsed, setIsCollapsed] = useState(false)

  return (
    <div
      className={cn(
        "relative h-full border-r border-border bg-sidebar flex flex-col",
        "transition-all duration-300 ease-in-out",
        isCollapsed ? "w-12" : "w-72"
      )}
    >
      {isCollapsed ? (
        <>
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setIsCollapsed(false)}
            className="absolute top-4 right-1 h-8 w-8 text-muted-foreground hover:text-foreground"
          >
            <ChevronRight className="h-4 w-4" />
          </Button>
          <div className="mt-12 flex flex-col items-center gap-4">
            <Button
              variant="ghost"
              size="icon"
              onClick={onNewChat}
              className="h-10 w-10 text-primary hover:bg-sidebar-accent"
            >
              <MessageSquarePlus className="h-5 w-5" />
            </Button>
          </div>
        </>
      ) : (
        <>
          {/* Collapse Button */}
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setIsCollapsed(true)}
            className="absolute top-4 right-2 h-8 w-8 text-muted-foreground hover:text-foreground z-10"
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>

          {/* Header */}
          <div className="p-4 pr-12">
            <div className="flex items-start gap-3">
              {/* Phoenix Icon */}
              <div className="relative flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-[#3D5A40] to-[#5A7A5E]">
                <svg width="28" height="28" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
                  {/* Phoenix body - interweaving lines */}
                  <path 
                    d="M16 28C16 28 12 24 12 18C12 14 14 10 16 8" 
                    stroke="#FFCC99" 
                    strokeWidth="2" 
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                  <path 
                    d="M16 28C16 28 20 24 20 18C20 14 18 10 16 8" 
                    stroke="#FFCC99" 
                    strokeWidth="2" 
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                  {/* Wings */}
                  <path 
                    d="M16 12C16 12 10 10 6 12C4 13 3 15 4 17C5 19 8 18 10 16C12 14 14 13 16 14" 
                    stroke="white" 
                    strokeWidth="1.8" 
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                  <path 
                    d="M16 12C16 12 22 10 26 12C28 13 29 15 28 17C27 19 24 18 22 16C20 14 18 13 16 14" 
                    stroke="white" 
                    strokeWidth="1.8" 
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                  {/* Head/flame */}
                  <path 
                    d="M16 8C16 8 14 5 16 3C18 5 16 8 16 8Z" 
                    fill="#FFCC99"
                    stroke="#FFCC99"
                    strokeWidth="1"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                  {/* Inner detail */}
                  <circle cx="16" cy="14" r="1.5" fill="#FFCC99" />
                </svg>
              </div>
              <div className="flex-1 min-w-0">
                <h1 className="text-base font-semibold text-[#3D5A40] leading-tight">高考辅导 AI 助手</h1>
                <div className="flex flex-wrap gap-1 mt-1.5">
                  <Badge variant="secondary" className="text-xs px-1.5 py-0 bg-[#3D5A40]/10 text-[#3D5A40] border-0">
                    学科答疑
                  </Badge>
                  <Badge variant="secondary" className="text-xs px-1.5 py-0 bg-[#FFCC99]/40 text-[#8B5A3C] border-0">
                    情绪支持
                  </Badge>
                  <Badge variant="secondary" className="text-xs px-1.5 py-0 bg-[#7A9E7E]/20 text-[#3D5A40] border-0">
                    计划制定
                  </Badge>
                </div>
              </div>
            </div>
          </div>

          {/* New Chat Button */}
          <div className="px-4 pb-4">
            <Button
              onClick={onNewChat}
              className="w-full justify-start gap-2 bg-[#3D5A40] hover:bg-[#4A6B4D] text-white"
            >
              <MessageSquarePlus className="h-4 w-4" />
              发起新对话
            </Button>
          </div>

          {/* Chat History */}
          <div className="flex-1 overflow-hidden flex flex-col">
            <div className="px-4 pb-2">
              <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">对话</span>
            </div>
            <ScrollArea className="flex-1 px-2">
              <div className="flex flex-col gap-1 pb-4">
                {chatHistory.map((chat) => (
                  <button
                    key={chat.id}
                    onClick={() => onSelectChat(chat.id)}
                    className={cn(
                      "w-full flex items-center gap-2 px-3 py-2 text-sm text-left rounded-lg transition-colors",
                      "hover:bg-sidebar-accent hover:text-sidebar-accent-foreground",
                      selectedChatId === chat.id 
                        ? "bg-sidebar-accent text-sidebar-accent-foreground" 
                        : "text-foreground"
                    )}
                  >
                    <MessageSquare className="h-4 w-4 flex-shrink-0 text-muted-foreground" />
                    <span className="truncate">{chat.title}</span>
                  </button>
                ))}
              </div>
            </ScrollArea>
          </div>

          {/* Settings & Help */}
          <div className="p-4 border-t border-border">
            <Button
              variant="ghost"
              className="w-full justify-start gap-2 text-muted-foreground hover:text-foreground"
            >
              <Settings className="h-4 w-4" />
              设置与帮助
            </Button>
          </div>
        </>
      )}
    </div>
  )
}
