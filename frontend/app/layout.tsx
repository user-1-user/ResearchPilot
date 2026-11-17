"use client";

import { useEffect, useState } from "react";
import "./globals.css";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Library, Network, PenTool, LayoutDashboard, LogOut, UserCircle, LogIn, Compass } from "lucide-react";
import LoginModal from "@/components/LoginModal";
import { apiFetch } from "@/lib/api";

export default function RootLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [user, setUser] = useState<string | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  useEffect(() => {
    const checkAuth = () => setUser(localStorage.getItem("yantu_user"));
    checkAuth();
    const handleOpenLogin = () => setIsModalOpen(true);
    window.addEventListener("open-login-modal", handleOpenLogin);
    window.addEventListener("storage", checkAuth);
    return () => {
      window.removeEventListener("open-login-modal", handleOpenLogin);
      window.removeEventListener("storage", checkAuth);
    };
  }, []);

  const handleLogout = async () => {
    try { await apiFetch("/api/auth/logout", { method: "POST" }); } catch { /* clear local credentials even if API is offline */ }
    localStorage.removeItem("yantu_token");
    localStorage.removeItem("yantu_user");
    setUser(null);
    window.location.reload();
  };

  const navItems = [
    { name: "工作台", href: "/", icon: LayoutDashboard },
    { name: "文献库", href: "/library", icon: Library },
    { name: "引用图谱", href: "/graph", icon: Network },
    { name: "论文发现", href: "/discover", icon: Compass },
    { name: "写作 Copilot", href: "/copilot", icon: PenTool },
  ];

  return (
    <html lang="zh-CN">
      <body className="flex h-screen overflow-hidden bg-slate-50 font-sans text-slate-900">
        <aside className="z-20 flex w-60 shrink-0 flex-col border-r border-slate-200 bg-white shadow-sm">
          <Link href="/" className="flex h-16 items-center gap-3 border-b border-slate-100 px-6">
            <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-600 text-sm font-bold text-white">研</span>
            <span className="text-lg font-bold tracking-tight text-blue-700">研途智研</span>
          </Link>
          <nav className="flex-1 space-y-2 p-4">
            {navItems.map((item) => {
              const Icon = item.icon;
              const active = item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
              return <Link key={item.href} href={item.href} className={`flex items-center gap-3 rounded-xl px-4 py-3 text-sm font-medium transition ${active ? "bg-blue-50 text-blue-700" : "text-slate-600 hover:bg-slate-50"}`}><Icon className="h-5 w-5" />{item.name}</Link>;
            })}
          </nav>
          <div className="border-t border-slate-100 p-4 text-[11px] text-slate-400">ResearchPilot · AI for Science</div>
        </aside>

        <div className="relative flex min-w-0 flex-1 flex-col overflow-hidden">
          <header className="z-10 flex h-16 shrink-0 items-center justify-end border-b border-slate-200 bg-white px-6 shadow-sm md:px-8">
            {user ? <div className="flex items-center gap-4"><span className="flex items-center gap-2 text-sm font-medium text-slate-700"><UserCircle className="h-5 w-5 text-blue-600" />{user}</span><button onClick={() => void handleLogout()} className="flex items-center gap-1 text-sm text-slate-500 transition hover:text-rose-600"><LogOut className="h-4 w-4" />退出</button></div> : <button onClick={() => setIsModalOpen(true)} className="flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-blue-700"><LogIn className="h-4 w-4" />登录 / 注册</button>}
          </header>
          <main className="min-h-0 flex-1 overflow-auto">{children}</main>
        </div>
        <LoginModal isOpen={isModalOpen} onClose={() => setIsModalOpen(false)} />
      </body>
    </html>
  );
}
