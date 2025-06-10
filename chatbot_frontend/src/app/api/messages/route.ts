import { promptMessageSchema } from "@/features/chat/schemas/prompt-message-schema";
import { prisma } from "@/lib/prisma";
import { auth } from "@clerk/nextjs/server";
import { notFound, redirect } from "next/navigation";
import { NextRequest, NextResponse } from "next/server";
import _ from "lodash";

export async function POST(req: NextRequest) {
  const url = new URL(req.url);
  const chatId = url.searchParams.get("chatId");
  const { message, language } = await req.json();
  console.log("message: ", message);
  try {
    const { userId } = await auth();

    if (!userId) {
      redirect("/login");
    }

    const validatedMessage = promptMessageSchema.parse(message);
    console.log("validateM: ", validatedMessage);
    const chat = chatId
      ? await prisma.chat.findUnique({
          where: { id: chatId },
          include: { messages: true },
        })
      : await prisma.chat.create({
          data: {
            userId,
            title: validatedMessage.content,
          },
          include: { messages: true },
        });

    if (!chat) {
      notFound();
    }

    const userMessage = await prisma.message.create({
      data: { chatId: chat.id, role: "user", ...validatedMessage },
    });
    console.log("user mess: ", userMessage)
    const messages = chat.messages.concat(userMessage);
    console.log(language)
    const response = await fetch("http://localhost:8000/process", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages, id: chat.id, language }),
    });

    if (!response.ok) {
      throw new Error("Failed to fetch from chatbot API");
    }

    const { answer } = await response.json();

    const botMessage = await prisma.message.create({
      data: {
        chatId: chat.id,
        role: "assistant",
        content: answer,
      },
    });

    return NextResponse.json(botMessage);
  } catch (error) {
    console.log(error);
    return NextResponse.json({ error }, { status: 500 });
  }
}
