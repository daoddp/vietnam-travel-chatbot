import { Chat, Message } from "@prisma/client";

export interface MessageClient
  extends Omit<Message, "id" | "chatId" | "createdAt" | "updatedAt"> {}

export interface ChatClient
  extends Omit<Chat, "userId" | "createdAt" | "updatedAt"> {}
