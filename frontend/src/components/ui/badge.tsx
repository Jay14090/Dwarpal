import { cva, type VariantProps } from "class-variance-authority";
import * as React from "react";
import { cn } from "@/lib/utils";

const badgeVariants = cva("inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-xs font-medium", {
  variants: {
    variant: {
      default: "border-transparent bg-primary/15 text-primary",
      secondary: "border-transparent bg-secondary text-secondary-foreground",
      outline: "text-foreground",
      resident: "border-resident/40 bg-resident/15 text-resident",
      staff: "border-staff/40 bg-staff/15 text-staff",
      unknown: "border-unknown/40 bg-unknown/15 text-unknown",
      pending: "border-pending/40 bg-pending/15 text-pending",
      high: "border-unknown/40 bg-unknown/20 text-unknown",
      critical: "border-unknown bg-unknown text-white",
      medium: "border-yellow-500/40 bg-yellow-500/15 text-yellow-400",
      low: "border-transparent bg-secondary text-muted-foreground",
    },
  },
  defaultVariants: { variant: "default" },
});

export type BadgeVariant = NonNullable<VariantProps<typeof badgeVariants>["variant"]>;

export function Badge({
  className,
  variant,
  ...props
}: React.ComponentProps<"span"> & VariantProps<typeof badgeVariants>) {
  return <span className={cn(badgeVariants({ variant }), className)} {...props} />;
}

const ROLE_VARIANTS = ["resident", "staff", "unknown", "pending"] as const;

export function RoleBadge({ role }: { role: string | null | undefined }) {
  const r = (ROLE_VARIANTS as readonly string[]).includes(role ?? "") ? (role as BadgeVariant) : "pending";
  return <Badge variant={r}>{role ?? "pending"}</Badge>;
}

export function SeverityBadge({ severity }: { severity: string }) {
  const v = (["low", "medium", "high", "critical"] as const).find((s) => s === severity) ?? "low";
  return <Badge variant={v}>{severity}</Badge>;
}
