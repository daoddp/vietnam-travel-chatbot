"use client";

import { SignIn } from "@clerk/nextjs";

export default function LoginPage() {
  return (
    <div className="flex items-center justify-center h-screen">
      <SignIn routing="path" path="/login" />{" "}
      {/* Giao diện đăng nhập của Clerk */}
    </div>
  );
}
