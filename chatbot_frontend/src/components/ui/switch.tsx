"use client";

import React from "react";
import Switch from "react-switch";

interface ThemeSwitchProps {
  theme: "light" | "dark";
  toggleTheme: () => void;
}

export const ThemeSwitch: React.FC<ThemeSwitchProps> = ({
  theme,
  toggleTheme,
}) => {
  return (
    <div className="flex items-center space-x-2">
      <span className="text-sm text-gray-500">
        {theme === "light" ? "Light Mode" : "Dark Mode"}
      </span>
      <Switch
        onChange={toggleTheme}
        checked={theme === "dark"}
        offColor="#f4f4f5" // Màu nền khi ở chế độ sáng
        onColor="#4B5563" // Màu nền khi ở chế độ tối
        offHandleColor="#d1d5db" // Màu nút khi ở chế độ sáng
        onHandleColor={theme === "dark" ? "#facc15" : "#ffffff"} // Màu nút vàng khi ở chế độ tối
        checkedIcon={false} // Tắt icon khi bật
        uncheckedIcon={false} // Tắt icon khi tắt
        height={20} // Chiều cao switch
        width={40} // Chiều rộng switch
      />
    </div>
  );
};
