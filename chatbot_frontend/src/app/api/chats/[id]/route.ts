import { prisma } from "@/lib/prisma";
import { auth } from "@clerk/nextjs/server";
import { notFound, redirect } from "next/navigation";
import { NextRequest, NextResponse } from "next/server";

export async function GET(
  req: NextRequest,
  { params }: { params: { id: string } },
) {
  const id = params.id;

  try {
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

    return NextResponse.json(chat);
  } catch (error) {
    return NextResponse.json({ error }, { status: 500 });
  }
}
