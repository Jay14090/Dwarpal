"use client";

import { Activity, BarChart3, Bell, Search, Shield, UserCog, Users } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { setActor } from "@/lib/api";
import { useActor } from "@/lib/use-actor";
import { cn } from "@/lib/utils";

const LINKS = [
  { href: "/live", label: "Live", icon: Activity },
  { href: "/search", label: "Search", icon: Search },
  { href: "/events", label: "Events", icon: Bell },
  { href: "/registry", label: "Registry", icon: Users },
  { href: "/metrics", label: "Metrics", icon: BarChart3 },
];

export function Nav() {
  const path = usePathname();
  const actor = useActor();
  return (
    <nav className="flex w-full shrink-0 items-center gap-1 overflow-x-auto border-b bg-card px-3 py-2 md:h-screen md:w-48 md:flex-col md:items-stretch md:border-b-0 md:border-r md:py-4">
      <div className="mr-3 flex items-center gap-2 px-2 font-mono text-sm font-bold tracking-widest text-primary md:mb-6 md:mr-0">
        <Shield className="size-5" /> DWARPAL
      </div>
      {LINKS.map(({ href, label, icon: Icon }) => (
        <Link
          key={href}
          href={href}
          className={cn(
            "flex items-center gap-2 rounded-md px-3 py-2 text-sm text-muted-foreground transition-colors hover:bg-accent hover:text-foreground",
            path.startsWith(href) && "bg-accent text-foreground",
          )}
        >
          <Icon className="size-4" />
          {label}
        </Link>
      ))}
      <label className="ml-auto flex items-center gap-1.5 px-2 text-xs text-muted-foreground md:mb-1 md:ml-0 md:mt-auto" title="Acting as (audit log). 'admin' may unblur faces.">
        <UserCog className="size-4 shrink-0" />
        <select
          value={actor === "admin" ? "admin" : "operator"}
          onChange={(e) => setActor(e.target.value)}
          className="rounded border bg-card px-1 py-0.5 text-xs text-foreground"
        >
          <option value="operator">operator</option>
          <option value="admin">admin</option>
        </select>
      </label>
    </nav>
  );
}
