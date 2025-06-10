import { auth } from "@clerk/nextjs/server";
import { notFound, redirect } from "next/navigation";
import { prisma } from "@/lib/prisma";
import { ChatProvider } from "@/features/chat/hooks/use-chat";
import { Chat } from "@/features/chat/components/chat";
import { Sidebar } from "@/features/chat/components/sidebar";

export default async function WithIdChatPage({
  params,
}: {
  params: { id: string };
}) {
  const id = params.id;
  const { userId } = await auth();

  if (!userId) {
    redirect("/login");
  }

  const chat = await prisma.chat.findUnique({
    where: { id },
    include: {
      messages: true,
    },
  });

  if (!chat || chat.userId !== userId) {
    notFound();
  }

  return (
    <ChatProvider chat={chat}>
      <div className="flex w-full h-screen">
        <Sidebar />
        <Chat />
      </div>
    </ChatProvider>
  );
}
