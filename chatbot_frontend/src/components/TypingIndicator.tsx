// TypingIndicator.tsx
import React from "react";

export const TypingIndicator = () => {
  return (
    <div className="inline-block p-3 rounded-2xl bg-gray-300 text-black shadow-md flex items-center justify-center">
      <div className="flex space-x-1">
        <span className="dot"></span>
        <span className="dot"></span>
        <span className="dot"></span>
      </div>
      <style jsx>{`
        .dot {
          display: inline-block;
          width: 8px;
          height: 8px;
          background-color: #333;
          border-radius: 50%;
          animation: bounce 1.2s infinite;
        }
        .dot:nth-child(1) {
          animation-delay: 0s;
        }
        .dot:nth-child(2) {
          animation-delay: 0.2s;
        }
        .dot:nth-child(3) {
          animation-delay: 0.4s;
        }
        @keyframes bounce {
          0%,
          80%,
          100% {
            transform: scale(0);
          }
          40% {
            transform: scale(1);
          }
        }
      `}</style>
    </div>
  );
};
