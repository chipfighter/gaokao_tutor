"use client"

import { useState, useRef, useEffect } from "react"
import { Send, Bot, User, Plus, SlidersHorizontal, Mic } from "lucide-react"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { cn } from "@/lib/utils"

export interface Message {
  id: string
  role: "user" | "assistant"
  content: string
}

interface ChatAreaProps {
  messages: Message[]
  onSendMessage: (content: string) => void
  isLoading?: boolean
}

export function ChatArea({ messages, onSendMessage, isLoading }: ChatAreaProps) {
  const [input, setInput] = useState("")
  const scrollAreaRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    // Scroll to bottom when messages change
    if (scrollAreaRef.current) {
      const scrollContainer = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]')
      if (scrollContainer) {
        scrollContainer.scrollTop = scrollContainer.scrollHeight
      }
    }
  }, [messages])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (input.trim() && !isLoading) {
      onSendMessage(input.trim())
      setInput("")
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  return (
    <div className="flex-1 flex flex-col h-full bg-background">
      {/* Messages Area */}
      <ScrollArea ref={scrollAreaRef} className="flex-1 px-8">
        <div className="max-w-3xl mx-auto py-6 flex flex-col gap-6">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-[60vh] text-center">
              {/* Phoenix Icon for Empty State */}
              <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-[#3D5A40] to-[#5A7A5E] mb-4">
                <svg width="36" height="36" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
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
                  <path 
                    d="M16 8C16 8 14 5 16 3C18 5 16 8 16 8Z" 
                    fill="#FFCC99"
                    stroke="#FFCC99"
                    strokeWidth="1"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                  <circle cx="16" cy="14" r="1.5" fill="#FFCC99" />
                </svg>
              </div>
              <h2 className="text-xl font-semibold text-[#3D5A40] mb-2">高考辅导 AI 助手</h2>
              <p className="text-muted-foreground max-w-md leading-relaxed">
                我是你的高考学习伙伴，可以帮助你解答学科问题、制定学习计划、提供情绪支持。有什么想问的吗？
              </p>
            </div>
          ) : (
            messages.map((message) => (
              <MessageBubble key={message.id} message={message} />
            ))
          )}
          {isLoading && (
            <div className="flex items-start gap-3">
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-[#3D5A40]/10 text-[#3D5A40] flex-shrink-0">
                <Bot className="h-4 w-4" />
              </div>
              <div className="bg-white border border-[#C8D6C9] rounded-2xl rounded-tl-sm px-4 py-3">
                <div className="flex items-center gap-1">
                  <span className="w-2 h-2 bg-[#3D5A40]/60 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                  <span className="w-2 h-2 bg-[#3D5A40]/60 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                  <span className="w-2 h-2 bg-[#3D5A40]/60 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
                </div>
              </div>
            </div>
          )}
        </div>
      </ScrollArea>

      {/* Input Area - Gemini Style with new palette */}
      <div className="bg-background px-8 py-4">
        <form onSubmit={handleSubmit} className="max-w-3xl mx-auto">
          <div className="bg-[#F5F3E8] rounded-3xl overflow-hidden border border-[#E8E5D8]">
            {/* Text Area at Top */}
            <div className="px-4 pt-4 pb-2">
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="输入你的问题..."
                rows={2}
                className={cn(
                  "w-full resize-none bg-transparent",
                  "text-sm text-foreground placeholder:text-muted-foreground",
                  "focus:outline-none",
                  "min-h-[60px] max-h-[200px]"
                )}
              />
            </div>
            
            {/* Toolbar at Bottom */}
            <div className="flex items-center px-3 pb-3 gap-1">
              {/* Left Side: Plus and Tools */}
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="h-9 w-9 rounded-full text-muted-foreground hover:text-[#3D5A40] hover:bg-white/50"
              >
                <Plus className="h-5 w-5" />
              </Button>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className="h-9 rounded-full text-muted-foreground hover:text-[#3D5A40] hover:bg-white/50 gap-1.5 px-3"
              >
                <SlidersHorizontal className="h-4 w-4" />
                <span className="text-sm">工具</span>
              </Button>
              
              {/* Flexible Spacer */}
              <div className="flex-1" />
              
              {/* Right Side: Mic, Send */}
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="h-9 w-9 rounded-full text-muted-foreground hover:text-[#3D5A40] hover:bg-white/50"
              >
                <Mic className="h-5 w-5" />
              </Button>
              <Button
                type="submit"
                size="icon"
                disabled={!input.trim() || isLoading}
                className={cn(
                  "h-9 w-9 rounded-full",
                  "bg-[#3D5A40] hover:bg-[#4A6B4D] text-white",
                  "disabled:opacity-50 disabled:cursor-not-allowed"
                )}
              >
                <Send className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </form>
      </div>
    </div>
  )
}

function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === "user"

  return (
    <div className={cn("flex items-start gap-3", isUser && "flex-row-reverse")}>
      <div className={cn(
        "flex h-8 w-8 items-center justify-center rounded-full flex-shrink-0",
        isUser ? "bg-[#3D5A40] text-white" : "bg-[#3D5A40]/10 text-[#3D5A40]"
      )}>
        {isUser ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
      </div>
      <div className={cn(
        "max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed",
        isUser 
          ? "bg-[#3D5A40] text-white rounded-tr-sm" 
          : "bg-white border border-[#C8D6C9] text-[#2D2D2D] rounded-tl-sm"
      )}>
        <div className="whitespace-pre-wrap">{message.content}</div>
      </div>
    </div>
  )
}
