"use client"

import { useState, useCallback } from "react"
import { LeftSidebar } from "@/components/left-sidebar"
import { RightPanel, NodeEvent, LogEntry } from "@/components/right-panel"
import { ChatArea, Message } from "@/components/chat-area"

const initialChatHistory: any[] = []

function timestamp(): string {
  return new Date().toLocaleTimeString("en-GB", { hour12: false })
}

export default function Home() {
  const [chatHistory, setChatHistory] = useState(initialChatHistory)
  const [selectedChatId, setSelectedChatId] = useState<string | undefined>()
  const [messages, setMessages] = useState<Message[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [logs, setLogs] = useState<LogEntry[]>([
    { type: "info", message: "[INFO] System initialized.", ts: timestamp() },
  ])
  const [nodeEvents, setNodeEvents] = useState<NodeEvent[]>([])

  const handleNewChat = useCallback(() => {
    setSelectedChatId(undefined)
    setMessages([])
    setNodeEvents([])
    setLogs([{ type: "info", message: "[INFO] New chat session started.", ts: timestamp() }])
  }, [])

  const handleSelectChat = useCallback((id: string) => {
    setSelectedChatId(id)
    setMessages([])
    setNodeEvents([])
  }, [])

  const handleSendMessage = useCallback(async (content: string) => {
    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content,
    }

    setMessages((prev) => [...prev, userMessage])
    setNodeEvents([])
    setLogs((prev) => [
      ...prev,
      { type: "info" as const, message: `[INFO] User query: ${content.slice(0, 60)}`, ts: timestamp() },
    ])

    setIsLoading(true)

    // ==== The only frontend-backend interaction: LangGraph streaming output (SSE) ====
    try {
      const response = await fetch("http://localhost:8000/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: content }),
      })

      if (!response.body) throw new Error("No response body")

      const reader = response.body.getReader()
      const decoder = new TextDecoder()

      // Create an empty assistant message placeholder
      const assistantMessageId = (Date.now() + 1).toString()
      setMessages((prev) => [
        ...prev,
        { id: assistantMessageId, role: "assistant", content: "" },
      ])

      let buffer = ""

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const parts = buffer.split("\n\n")
        buffer = parts.pop() || ""

        for (const part of parts) {
          if (part.startsWith("data: ")) {
            const dataStr = part.slice(6)
            try {
              const data = JSON.parse(dataStr)

              // -- Token: append text to assistant message --
              if (data.type === "token") {
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === assistantMessageId
                      ? { ...msg, content: msg.content + data.content }
                      : msg
                  )
                )
              }

              // -- Node lifecycle event: update reasoning path + logs --
              if (data.type === "node_event") {
                const node: string = data.node
                const status: "start" | "end" = data.status
                const now = timestamp()

                // Update reasoning path state
                setNodeEvents((prev) => {
                  if (status === "start") {
                    return [...prev, { node, status: "running", ts: now }]
                  }
                  // Mark matching node as done
                  return prev.map((e) =>
                    e.node === node && e.status === "running"
                      ? { ...e, status: "done", endTs: now }
                      : e
                  )
                })

                // Append to system logs
                const label = status === "start" ? "Entering" : "Leaving"
                setLogs((prev) => [
                  ...prev,
                  { type: "info", message: `[INFO] ${label} node: ${node}`, ts: now },
                ])
              }
            } catch {
              // Ignore partial or malformed JSON chunks
            }
          }
        }
      }

      // Stream complete
      setLogs((prev) => [
        ...prev,
        { type: "info", message: "[INFO] Stream complete.", ts: timestamp() },
      ])
    } catch (error: any) {
      setLogs((prev) => [
        ...prev,
        { type: "error", message: `[ERROR] ${error.message}`, ts: timestamp() },
      ])
    } finally {
      setIsLoading(false)

      if (!selectedChatId) {
        const newChat = {
          id: Date.now().toString(),
          title: content.slice(0, 30) + (content.length > 30 ? "..." : ""),
        }
        setChatHistory((prev) => [newChat, ...prev])
        setSelectedChatId(newChat.id)
      }
    }
  }, [selectedChatId])

  return (
    <div className="flex h-screen overflow-hidden">
      <LeftSidebar
        chatHistory={chatHistory}
        onNewChat={handleNewChat}
        onSelectChat={handleSelectChat}
        selectedChatId={selectedChatId}
      />
      <ChatArea
        messages={messages}
        onSendMessage={handleSendMessage}
        isLoading={isLoading}
      />
      <RightPanel logs={logs} nodeEvents={nodeEvents} />
    </div>
  )
}
