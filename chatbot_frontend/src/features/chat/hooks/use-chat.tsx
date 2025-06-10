"use client";

import { Chat, Message, Prisma } from "@prisma/client";
import { MessageClient, ChatClient } from "../types/message";

import React, {
  createContext,
  useContext,
  useState,
  ReactNode,
  FormEvent,
  useEffect,
} from "react";

type WithMessagesChat = Prisma.ChatGetPayload<{ include: { messages: true } }>;

interface ChatContextType {
  messages: MessageClient[];
  setMessages: React.Dispatch<React.SetStateAction<MessageClient[]>>;
  input: string;
  setInput: React.Dispatch<React.SetStateAction<string>>;
  history: ChatClient[];
  setHistory: React.Dispatch<React.SetStateAction<ChatClient[]>>;
  selectedChat: string | null;
  setSelectedChat: React.Dispatch<React.SetStateAction<string | null>>;
  sidebarOpen: true | false;
  setSidebarOpen: React.Dispatch<React.SetStateAction<true | false>>;
  isTyping: true | false;
  setIsTyping: React.Dispatch<React.SetStateAction<true | false>>;
  theme: "light" | "dark";
  language: "Tiếng Việt" | "English";
  setLanguage: React.Dispatch<React.SetStateAction<"Tiếng Việt" | "English">>;
  handleLanguageChange: (event: React.ChangeEvent<HTMLSelectElement>) => void;
  toggleTheme: () => void;
  toggleSidebar: () => void;
  startNewChat: () => void;
  handleSubmit: (event: FormEvent) => Promise<void>;
  handleSuggestionClick: (question: string) => void;
}

const ChatContext = createContext<ChatContextType | undefined>(undefined);

export const ChatProvider = ({
  chat,
  children,
}: {
  chat?: WithMessagesChat;
  children: ReactNode;
}) => {
  const [messages, setMessages] = useState<MessageClient[]>([]);
  const [input, setInput] = useState("");
  const [history, setHistory] = useState<ChatClient[]>([]);
  const [selectedChat, setSelectedChat] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [isTyping, setIsTyping] = useState(false);
  const [language, setLanguage] = useState<"Tiếng Việt" | "English">("Tiếng Việt");
  const [theme, setTheme] = useState<"light" | "dark">(() => {
    if (typeof window !== "undefined") {
      const savedTheme = localStorage.getItem("theme");
      return savedTheme ? (savedTheme as "light" | "dark") : "light";
    }
    return "light";
  });

  useEffect(() => {
    if (chat) {
      setMessages(chat.messages);
      setSelectedChat(chat.id);
    }
  }, [chat]);

  const toggleTheme = () => {
    const newTheme = theme === "light" ? "dark" : "light";
    setTheme(newTheme);
    // Lưu theme vào localStorage
    if (typeof window !== "undefined") {
      localStorage.setItem("theme", newTheme);
    }
  };

  const toggleSidebar = () => {
    setSidebarOpen((prev) => !prev); // Toggle trạng thái mở/đóng của sidebar
  };

  const startNewChat = () => {
    setSelectedChat(null);
    setMessages([]);
  };

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    if (!input.trim()) return;

    const newMessage: MessageClient = { role: "user", content: input };
    setMessages((prevMessages) => [...prevMessages, newMessage]);
    setInput("");
    setIsTyping(true)

    const url = selectedChat
      ? `/api/messages?chatId=${selectedChat}`
      : `/api/messages`;
    try {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({message: newMessage, language}),
      });

      if (!response.ok) throw new Error("Failed to fetch from API");
      const { content, chatId } = await response.json();

      const botMessage: MessageClient = { role: "assistant", content: content };

      setIsTyping(false)
      setMessages((prevMessages) => [...prevMessages, botMessage]);
      setHistory((prevHistory) => {
        const updatedHistory = prevHistory.map((chat) =>
          chat.id === chatId
            ? {
                ...chat,
                // messages: [...(chat as ChatClient & { messages: MessageClient[] }).messages, newMessage, botMessage],
              }
            : chat,
        );
        if (!prevHistory.find((chat) => chat.id === chatId)) {
          updatedHistory.push({
            id: chatId,
            title: newMessage.content,
            messages: [newMessage, botMessage],
          } as ChatClient);
        }
        return updatedHistory;
      });
    } catch (error) {
      console.error("Error occurred:", error);
    }
  };

  const handleSuggestionClick = async (question: string) => {
    if (!question.trim()) return;

    const newMessage: MessageClient = { role: "user", content: question };
    setMessages((prevMessages) => [...prevMessages, newMessage]);
    setInput("");
    setIsTyping(true)

    try {
      const response = await fetch("/api/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({message: newMessage, language}),
      });

      if (!response.ok) throw new Error("Failed to fetch from API");
      const { content, chatId } = await response.json();
      console.log(chatId);
      const botMessage: MessageClient = { role: "assistant", content: content };

      setIsTyping(false)
      setMessages((prevMessages) => [...prevMessages, botMessage]);
      setHistory((prevHistory) => {
        const updatedHistory = prevHistory.map((chat) =>
          chat.id == chatId
            ? {
                ...chat,
              }
            : chat,
        );
        // if (!prevHistory.find((chat) => chat.id === chatId)) {
        updatedHistory.push({
          id: chatId,
          title: newMessage.content,
          messages: [newMessage, botMessage],
        } as ChatClient);
        // }
        return updatedHistory;
      });
    } catch (error) {
      console.error("Error occurred:", error);
    }
  };

  const handleLanguageChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const selectedLanguage = event.target.value as "Tiếng Việt" | "English";
    setLanguage(selectedLanguage);
  };  

  return (
    <ChatContext.Provider
      value={{
        messages,
        setMessages,
        input,
        setInput,
        history,
        setHistory,
        selectedChat,
        setSelectedChat,
        theme,
        toggleTheme,
        startNewChat,
        handleSuggestionClick,
        handleSubmit,
        toggleSidebar,
        sidebarOpen,
        setSidebarOpen,
        isTyping,
        setIsTyping,
        language,
        setLanguage,
        handleLanguageChange,
      }}
    >
      {children}
    </ChatContext.Provider>
  );
};

export const useChat = () => {
  const context = useContext(ChatContext);
  if (!context) {
    throw new Error("useChat must be used within a ChatProvider");
  }
  return context;
};
