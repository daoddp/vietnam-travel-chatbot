"use client";

import { FolderIcon, PlusCircleIcon, ArrowLeftIcon, InboxIcon, ChatBubbleBottomCenterTextIcon } from "@heroicons/react/24/solid";
import { useChat } from "../hooks/use-chat";
import Link from "next/link";
import { useEffect } from "react";
import { Button } from "@/components/ui/button";
export const Sidebar = () => {
  const {
    theme,
    setHistory,
    history,
    selectedChat,
    startNewChat,
    toggleSidebar,
    sidebarOpen,
    language,
  } = useChat();

  // Lấy danh sách lịch sử từ backend
  useEffect(() => {
    fetch("/api/chats")
      .then((res) => res.json())
      .then((data) => {
        setHistory(data);
      })
      .catch((err) => console.error("Error fetching chat history:", err));
  }, []);
  return (
    <>
      {sidebarOpen && (
        <aside
          className={`relative flex-shrink-0 max-w-xs border-r 
                      ${theme === "light" 
                        ? "bg-gray-200 text-black" 
                        : "bg-gray-800"} 
                        p-4 pt-10 overflow-visible`}
          style={{
            lineHeight: "30px",
            fontSize: "17,5px",
            tabSize: "4",
            width: "600px",
            overflow: "hidden",
            overflowY: "auto",
            /* CSS tùy chỉnh cho thanh cuộn */
            scrollbarColor:
              theme === "light"
                ? "#000000 #f0f0f0"
                : "#ffffff #000000" /* Thumb màu đen, Track màu sáng */,
            scrollbarWidth: "thin" /* Chỉnh kích thước thanh cuộn */,
          }}
        >
          {/* Container cho các nút */}
          <div className="absolute top-4 left-4 right-4 z-10 flex justify-between items-center overflow-visible">
            {/* Nút New Chat */}
            <div className="relative group overflow-visible">
              <Button
                onClick={startNewChat}
                className={`p-2 rounded-full transition-colors duration-200
                            ${theme === "light"
                            ? "hover:bg-blue-600 text-black"
                            : "hover:bg-yellow-400 text-white"}`}>
                <PlusCircleIcon className="w-8 h-8" />
              </Button>

              {/* Tooltip */}
              <div className="absolute left-1/2 -translate-x-1/2 mt-2 w-max 
                              p-1 text-sm text-white bg-black rounded opacity-0 
                              group-hover:opacity-100 transition-opacity duration-200 overflow-visible">
                {language === "Tiếng Việt" ? "Cuộc trò chuyện mới" : "New Chat"}
              </div>
            </div>

            {/* Nút Close Sidebar */}
            <div className="relative group">
              <Button
                onClick={toggleSidebar}
                className={`p-2 rounded-full transition-colors duration-200 overflow-visible
                            ${theme === "light"
                            ? "hover:bg-blue-600 text-black"
                            : "hover:bg-yellow-400 text-white"}`}>
                <ArrowLeftIcon className="w-7 h-7" />
              </Button>

              {/* Tooltip */}
              <div className="absolute top-full right-1/2 translate-x-1/2 
                              translate-y-2 w-max p-1 text-sm text-white bg-black 
                              rounded opacity-0 group-hover:opacity-100 transition-opacity 
                              duration-200 shadow-lg z-20 overflow-visible">
                {language === "Tiếng Việt" ? "Đóng thanh bên" : "Close Sidebar"}
              </div>
            </div>
          </div>

          {/* button new chat */}
          <Link href="/">
            <Button
              onClick={startNewChat}
              className={`group w-full max-w-sm px-4 py-2 mb-4 mt-14 rounded-full flex items-center justify-between gap-2 transition-colors duration-200
                ${theme === "light" ? "text-black bg-gray-300 hover:bg-blue-600" : "text-black bg-yellow-400 hover:bg-yellow-500"} 
                text-lg font-bold`}
            >
              <div className="flex items-center gap-5">
                <ChatBubbleBottomCenterTextIcon className="w-7 h-7" />
                {language === "Tiếng Việt" ? "Trò chuyện ngay!" : "Chat now!"}
              </div>
              {/* Icon hiển thị khi hover */}
              <PlusCircleIcon className="w-6 h-6 opacity-0 group-hover:opacity-100 transition-opacity duration-200" />
            </Button>
          </Link>
          {/* chat history */}
          <h1 className="max-w-sm mb-4 mt-7 text-lg font-bold flex items-center gap-2">
            <FolderIcon className="w-5 h-5" />
            {language === "Tiếng Việt" ? "Lịch sử trò chuyện" : "Chat history"}
          </h1>
          

          {history.length === 0 ? (
            <p className="text-center text-muted-foreground">
              Không tồn tại lịch sử trò chuyện
            </p>
          ) : (
            <ul>
              {history.map((chat) => (
                <Link href={`/chat/${chat.id}`}>
                  <li
                    key={chat.id}
                    className={`p-2 rounded-2xl cursor-pointer transition-colors duration-200
                                    ${
                                      selectedChat === chat.id
                                        ? theme === "light"
                                          ? "bg-gray-300"
                                          : "bg-gray-700"
                                        : theme === "light"
                                          ? "hover:bg-blue-600"
                                          : "hover:bg-yellow-400"
                                    }`}
                    style={{
                      display: "block",
                      whiteSpace: "nowrap",
                      overflow: "hidden",
                      textOverflow: "clip", // Không dùng dấu "..."
                      maxWidth: "590px",
                      maskImage:
                        "linear-gradient(to right, black 80%, transparent)", // Tạo hiệu ứng mờ dần
                      WebkitMaskImage:
                        "linear-gradient(to right, black 80%, transparent)", // Tương thích Webkit
                    }}
                  >
                    {chat.title}
                  </li>
                </Link>
              ))}
            </ul>
          )}
        </aside>
      )}
    </>
  );
};
