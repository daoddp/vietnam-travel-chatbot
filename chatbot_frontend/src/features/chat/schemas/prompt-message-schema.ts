import { MESSAGE_TEXT_MAX_LENGTH } from "@/constants";
import z from "zod";

export const promptMessageSchema = z.object({
  content: z.string().min(1).max(MESSAGE_TEXT_MAX_LENGTH),
});
