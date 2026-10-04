import type { Metadata } from "next";
import { Nav } from "@/components/nav";
import "./globals.css";

export const metadata: Metadata = {
  title: "Dwarpal",
  description: "AI security operator for gated communities",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark h-full antialiased">
      <body className="flex min-h-full flex-col md:flex-row">
        <Nav />
        <main className="min-w-0 flex-1 p-4 md:h-screen md:overflow-y-auto md:p-6">{children}</main>
      </body>
    </html>
  );
}
