import React, { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import rehypeSanitize from "rehype-sanitize";
import { MessageCircle, X, Send, Loader2 } from "lucide-react";
import hljs from "highlight.js/lib/core";
import javascript from "highlight.js/lib/languages/javascript";
import python from "highlight.js/lib/languages/python";

hljs.registerLanguage("javascript", javascript);
hljs.registerLanguage("python", python);


const ChatbotButton = () => {
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [message, setMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: "Hi! I'm AIVY, your healthcare assistant. Ask me anything about your patient journey.",
      sender: "bot",
    },
  ]);

  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleChatToggle = () => {
    setIsChatOpen(!isChatOpen);
  };

  const handleSendMessage = async () => {
    const trimmedMessage = message.trim();
    if (trimmedMessage === "") return;
  
    const newUserMessage = {
      id: Date.now(),
      text: trimmedMessage,
      sender: "user",
    };
  
    setMessages((prev) => [...prev, newUserMessage]);
    setMessage("");
    setIsLoading(true);
  
    try {
      const botResponseId = Date.now() + 1;
      const initialBotMessage = { id: botResponseId, text: "", sender: "bot" };
      setMessages((prev) => [...prev, initialBotMessage]);
  
      const response = await fetch("https://alz-backend-1.onrender.com/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: trimmedMessage }),
      });
  
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let fullResponse = "";
  
      while (true) {
        const { done, value } = await reader?.read();
        if (done) break;
  
        const chunk = decoder.decode(value);
        const events = chunk.split("\n\n");
  
        events.forEach((event) => {
          if (event.startsWith("data: ")) {
            const content = event.replace("data: ", "").trim();
  
            if (content === "[DONE]") {
              setIsLoading(false);
              return;
            }
  
            // Update with full response
            fullResponse = content;
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === botResponseId ? { ...msg, text: fullResponse } : msg
              )
            );
          }
        });
      }
    } catch (error) {
      console.error("Chat error:", error);
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 1,
          text: "Sorry, there was an issue connecting to the service. Please try again.",
          sender: "bot",
        },
      ]);
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !isLoading) {
      handleSendMessage();
    }
  };

  return (
    <>
      <div className="fixed bottom-6 right-6 z-50">
        <button
          onClick={handleChatToggle}
          className="bg-purple-600 text-white p-4 rounded-full shadow-2xl hover:bg-purple-700 transition-all duration-300 transform hover:scale-105 active:scale-95"
        >
          <MessageCircle className="w-6 h-6" />
        </button>
      </div>

      <div
        className={`fixed top-0 right-0 h-full w-[500px] bg-white shadow-2xl z-[100] transition-transform duration-300 ease-in-out ${
          isChatOpen ? "translate-x-0" : "translate-x-full"
        }`}
      >
        <div className="bg-purple-600 text-white p-4 flex justify-between items-center">
          <h3 className="font-semibold text-lg">AIVY - Patient Journey Assistant</h3>
          <button
            onClick={handleChatToggle}
            className="hover:bg-purple-700 p-1 rounded-full transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="flex-1 h-[calc(100%-200px)] overflow-y-auto p-4 space-y-3">
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex ${msg.sender === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[80%] p-3 rounded-xl break-words whitespace-pre-wrap ${
                msg.sender === "user" ? "bg-purple-100 text-right" : "bg-blue-100 text-left"
              }`}
            >
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                rehypePlugins={[rehypeRaw, rehypeSanitize]}
                components={{
                  strong: ({node, ...props}) => <strong className="font-bold" {...props} />,
                  li: ({node, ...props}) => (
                    <span className="block mb-2">
                      <span className="font-semibold">{props.children}</span>
                    </span>
                  ),
                  ol: ({node, ...props}) => (
                    <div className="space-y-1">
                      {props.children}
                    </div>
                  )
                }}
              >
                {msg.text}
              </ReactMarkdown>
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

        <div className="absolute bottom-0 left-0 right-0 p-4 border-t flex items-center gap-2">
          <input
            ref={inputRef}
            type="text"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask here..."
            disabled={isLoading}
            className="flex-1 p-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 disabled:opacity-50 disabled:cursor-not-allowed"
          />
          <button
            onClick={handleSendMessage}
            disabled={isLoading || message.trim() === ""}
            className="bg-purple-600 text-white p-2 rounded-full hover:bg-purple-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
          >
            {isLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Send className="w-5 h-5" />}
          </button>
        </div>
      </div>
    </>
  );
};

export default ChatbotButton;
