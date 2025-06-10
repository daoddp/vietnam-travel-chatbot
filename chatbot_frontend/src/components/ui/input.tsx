import * as React from "react";
import { cn } from "@/lib/utils";

export interface InputProps
  extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  onEnterPress?: () => void; // Hàm callback khi nhấn Enter
}

const Input = React.forwardRef<HTMLTextAreaElement, InputProps>(
  ({ className, onEnterPress, ...props }, ref) => {
    const [height, setHeight] = React.useState("44px");
    const [overflow, setOverflow] = React.useState<"hidden" | "auto">("hidden");

    const maxHeight = "150px";

    const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      const text = e.target.value;

      if (text.length === 0) {
        setHeight("44px");
        setOverflow("hidden");
      } else {
        setHeight("44px");

        const newHeight = Math.min(e.target.scrollHeight, parseInt(maxHeight));
        setHeight(`${newHeight}px`);

        if (newHeight >= parseInt(maxHeight)) {
          setOverflow("auto");
        } else {
          setOverflow("hidden");
        }
      }
    };

    const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      // Kiểm tra nếu phím Enter được nhấn mà không giữ Shift
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault(); // Ngừng hành động mặc định (thêm dòng mới)
        if (onEnterPress) {
          onEnterPress(); // Gọi hàm callback khi nhấn Enter
        }
      }
    };

    return (
      <textarea
        ref={ref}
        {...props}
        onInput={handleInput}
        onKeyDown={handleKeyDown} // Xử lý sự kiện keydown
        style={{ height, overflowY: overflow }}
        className={cn(
          "flex w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50",
          className,
        )}
      />
    );
  },
);

Input.displayName = "Input";

export { Input };
