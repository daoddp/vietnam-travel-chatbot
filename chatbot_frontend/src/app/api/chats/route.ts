import { prisma } from "@/lib/prisma";
import { auth } from "@clerk/nextjs/server";
import { redirect } from "next/navigation";
import { NextRequest, NextResponse } from "next/server";

export async function GET(req: NextRequest) {
  try {
    const { userId } = await auth();

    if (!userId) {
      redirect("/login");
    }

    const chats = await prisma.chat.findMany({ where: { userId } });

    return NextResponse.json(chats);
  } catch (error) {
    return NextResponse.json({ error }, { status: 500 });
  }
}
