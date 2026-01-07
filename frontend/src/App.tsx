import { useState, useEffect, useRef } from "react";
import { Menu, Sun, Moon } from "lucide-react";
import { Sidebar } from "./components/Sidebar";
import { Message } from "./components/Message";
import { ChatInput } from "./components/ChatInput";

interface MessageType {
  id: string;
  role: "user" | "assistant";
  content: string;
}

interface Conversation {
  id: string;
  title: string;
  date: string;
  messages: MessageType[];
}

export default function App() {
  const [isDark, setIsDark] = useState(true);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [conversations, setConversations] = useState<Conversation[]>([
    {
      id: "1",
      title: "New conversation",
      date: "Today",
      messages: [],
    },
  ]);
  const [activeConversationId, setActiveConversationId] = useState("1");
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const botAvatarUrl = "/bot-avatar.png";

  const activeConversation = conversations.find((c) => c.id === activeConversationId);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [activeConversation?.messages]);

  const handleSendMessage = async (content: string) => {
    const newUserMessage: MessageType = {
      id: Date.now().toString(),
      role: "user",
      content,
    };

    setConversations((prev) =>
      prev.map((conv) =>
        conv.id === activeConversationId
          ? {
              ...conv,
              messages: [...conv.messages, newUserMessage],
              title:
                conv.messages.length === 0
                  ? content.slice(0, 30) + (content.length > 30 ? "..." : "")
                  : conv.title,
            }
          : conv
      )
    );

    setIsTyping(true);

    try {
      const response = await fetch("/api/v1/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: content,
          history: [],
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: any = await response.json();

      let assistantContent: string = String(data.answer ?? "");


      const rawSources = data.sources ?? data.retrieved ?? [];

      if (Array.isArray(rawSources) && rawSources.length > 0) {
        assistantContent += "\n\n---\n**Sources:**\n";

        rawSources.forEach((src: any, index: number) => {
          const text =
            typeof src === "string" ? src : String(src.text ?? src.content ?? "");

          const cid =
            typeof src === "string"
              ? "unknown"
              : String(src.chunk_id ?? src.id ?? `source_${index + 1}`);

          const preview =
            text.slice(0, 300).replace(/\n/g, " ") + (text.length > 300 ? "..." : "");

          assistantContent += `${index + 1}. [${cid}] ${preview}\n`;
        });
      }

      const jr = data.judge_result ?? data.judge ?? null;
      if (jr) {
        const verdict = String(jr.verdict ?? "unknown").toUpperCase();
        assistantContent += `\n\n---\n**Judge Verdict:** ${verdict}`;
      }

      const assistantMessage: MessageType = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: assistantContent,
      };

      setConversations((prev) =>
        prev.map((conv) =>
          conv.id === activeConversationId
            ? { ...conv, messages: [...conv.messages, assistantMessage] }
            : conv
        )
      );
    } catch (error) {
      console.error("Error:", error);
      const errorMessage: MessageType = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content:
          "Sorry, I encountered an error connecting to the server. Please make sure the backend is running.",
      };

      setConversations((prev) =>
        prev.map((conv) =>
          conv.id === activeConversationId
            ? { ...conv, messages: [...conv.messages, errorMessage] }
            : conv
        )
      );
    } finally {
      setIsTyping(false);
    }
  };

  const handleNewChat = () => {
    const newConversation: Conversation = {
      id: Date.now().toString(),
      title: "New conversation",
      date: "Today",
      messages: [],
    };
    setConversations((prev) => [newConversation, ...prev]);
    setActiveConversationId(newConversation.id);
    setIsSidebarOpen(false);
  };

  const handleSelectConversation = (id: string) => {
    setActiveConversationId(id);
    setIsSidebarOpen(false);
  };

  return (
    <div className={`h-screen flex ${isDark ? "bg-[#343541]" : "bg-white"} transition-colors`}>
      <Sidebar
        isOpen={isSidebarOpen}
        onClose={() => setIsSidebarOpen(false)}
        conversations={conversations}
        activeConversation={activeConversationId}
        onSelectConversation={handleSelectConversation}
        onNewChat={handleNewChat}
        isDark={isDark}
      />

      <div className="flex-1 flex flex-col h-screen overflow-hidden">
        <header
          className={`flex items-center justify-between px-4 py-3 border-b ${
            isDark ? "border-white/10 bg-[#343541]" : "border-gray-200 bg-white"
          }`}
        >
          <div className="flex items-center gap-3">
            <button
              onClick={() => setIsSidebarOpen(!isSidebarOpen)}
              className={`lg:hidden p-2 rounded-lg transition-colors ${
                isDark ? "hover:bg-white/10 text-white" : "hover:bg-gray-100 text-gray-900"
              }`}
            >
              <Menu size={20} />
            </button>
            <h1 className={`font-semibold ${isDark ? "text-white" : "text-gray-900"}`}>
              ChudGTP
            </h1>
          </div>
          <button
            onClick={() => setIsDark(!isDark)}
            className={`p-2 rounded-lg transition-colors ${
              isDark ? "hover:bg-white/10 text-white" : "hover:bg-gray-100 text-gray-900"
            }`}
            aria-label="Toggle theme"
          >
            {isDark ? <Sun size={20} /> : <Moon size={20} />}
          </button>
        </header>

        <div className="flex-1 overflow-y-auto">
          {activeConversation?.messages.length === 0 ? (
            <div className="h-full flex items-center justify-center">
              <div className="text-center px-4">
                <div className="w-20 h-20 mx-auto mb-4 flex items-center justify-center">
                  <img
                    src={botAvatarUrl}
                    alt="ChudGTP"
                    className="w-full h-full rounded-full object-cover shadow-lg"
                  />
                </div>
                <h2 className={`text-2xl font-semibold mb-2 ${isDark ? "text-white" : "text-gray-900"}`}>
                  How can I help you today?
                </h2>
                <p className={`${isDark ? "text-gray-400" : "text-gray-600"}`}>
                  (Yes, it's ChudGTP not ChatGPT)
                </p>
              </div>
            </div>
          ) : (
            <div>
              {activeConversation?.messages.map((message) => (
                <Message
                  key={message.id}
                  role={message.role}
                  content={message.content}
                  isDark={isDark}
                />
              ))}

              {isTyping && (
                <div className={`w-full ${isDark ? "bg-[#444654]" : "bg-white"}`}>
                  <div className="max-w-3xl mx-auto px-4 py-6 flex gap-6">
                    <div className="flex-shrink-0">
                      <img
                        src={botAvatarUrl}
                        alt="ChudGTP"
                        className="w-8 h-8 rounded-full object-cover"
                      />
                    </div>
                    <div className="flex-1">
                      <div className="flex gap-1 mt-2">
                        <div
                          className={`w-2 h-2 rounded-full animate-bounce ${
                            isDark ? "bg-gray-400" : "bg-gray-600"
                          }`}
                          style={{ animationDelay: "0ms" }}
                        />
                        <div
                          className={`w-2 h-2 rounded-full animate-bounce ${
                            isDark ? "bg-gray-400" : "bg-gray-600"
                          }`}
                          style={{ animationDelay: "150ms" }}
                        />
                        <div
                          className={`w-2 h-2 rounded-full animate-bounce ${
                            isDark ? "bg-gray-400" : "bg-gray-600"
                          }`}
                          style={{ animationDelay: "300ms" }}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        <ChatInput onSend={handleSendMessage} isDark={isDark} disabled={isTyping} />
      </div>
    </div>
  );
}
