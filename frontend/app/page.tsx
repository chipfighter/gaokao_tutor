"use client"

import { useState, useCallback } from "react"
import { LeftSidebar } from "@/components/left-sidebar"
import { RightPanel } from "@/components/right-panel"
import { ChatArea, Message } from "@/components/chat-area"

const initialChatHistory: any[] = []

export default function Home() {
  const [chatHistory, setChatHistory] = useState(initialChatHistory)
  const [selectedChatId, setSelectedChatId] = useState<string | undefined>()
  const [messages, setMessages] = useState<Message[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [logs, setLogs] = useState([
    { type: "info" as const, message: "[INFO] System initialized." },
  ])

  const handleNewChat = useCallback(() => {
    setSelectedChatId(undefined)
    setMessages([])
    setLogs([{ type: "info" as const, message: "[INFO] New chat session started." }])
  }, [])

  const handleSelectChat = useCallback((id: string) => {
    setSelectedChatId(id)
    // In a real scenario, you can request the backend to get chat history here; for now, just clear it
    setMessages([])
  }, [])

  const handleSendMessage = useCallback(async (content: string) => {
    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content,
    }

    setMessages((prev) => [...prev, userMessage])
    setLogs((prev) => [
      ...prev,
      { type: "info" as const, message: `[INFO] User query: ${content}` },
    ])

    setIsLoading(true)

    // ==== The only frontend-backend interaction: LangGraph streaming output (SSE) ====
    try {
      // Make the actual cross-origin SSE API call here
      const response = await fetch("http://localhost:8000/stream", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          // Ensure the backend is configured with CORS to allow cross-origin requests
        },
        body: JSON.stringify({ query: content })
      });

      if (!response.body) throw new Error("No response body");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      // Create an empty assistant message placeholder
      const assistantMessageId = (Date.now() + 1).toString();
      setMessages((prev) => [
        ...prev,
        { id: assistantMessageId, role: "assistant", content: "" }
      ]);

      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop() || "";
        
        for (const part of parts) {
          if (part.startsWith("data: ")) {
            const dataStr = part.slice(6);
            try {
              const data = JSON.parse(dataStr);
              if (data.type === "token") {
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === assistantMessageId
                      ? { ...msg, content: msg.content + data.content }
                      : msg
                  )
                );
              }
            } catch (err) {
              // Ignore partial or malformed JSON chunks
            }
          }
        }
      }

    } catch (error: any) {
      setLogs((prev) => [
        ...prev,
        { type: "error" as const, message: `[ERROR] ${error.message}` },
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
      <RightPanel logs={logs} />
    </div>
  )
}
