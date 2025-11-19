import socket from "../../services/socket";
import * as React from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarInset,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarProvider,
  SidebarRail,
  SidebarTrigger,
} from "../ui/sidebar";
import { ScrollArea } from "../ui/scroll-area";
import { Avatar, AvatarFallback } from "../ui/avatar";
import { Users2, FolderCog, MapPinned, Layers, Settings, LogOut } from "lucide-react";

const NAV = [
  {
    label: "Manage Accounts",
    to: "/admin",
    icon: Users2,
    match: (p: string) => p === "/admin",
  },
  {
    label: "Manage Projects",
    to: "/admin/projects",
    icon: FolderCog,
    match: (p: string) => p.startsWith("/admin/projects"),
  },
  {
    label: "Manage Maps",
    to: "/admin/layers",
    icon: MapPinned,
    match: (p: string) => p.startsWith("/admin/layers"),
  },
  {
    label: "Manage Land Cover",
    to: "/admin/landcover",
    icon: Layers,
    match: (p: string) => p.startsWith("/admin/landcover"),
  },
];

export function AppShell({ children }: { children: React.ReactNode }) {
  return (
    <SidebarProvider>
      <AppSidebar />
      <SidebarInset>
        {/* Top bar */}
        <header className="flex h-14 items-center gap-2 border-b px-3">
          <SidebarTrigger />
          <span className="text-sm font-semibold">EcoSense • Admin</span>
        </header>

        {/* Page content */}
        <div className="flex-1 p-4 md:p-6">{children}</div>
      </SidebarInset>
    </SidebarProvider>
  );
}

export function AppSidebar() {
  const { pathname } = useLocation();
  const navigate = useNavigate();
  const [user, setUser] = React.useState<{ name: string; email: string } | null>(
    null
  );

  //Fetch logged-in user info
  React.useEffect(() => {
    const fetchUser = async () => {
      try {
        const res = await fetch(`${import.meta.env.VITE_API_URL}/auth/me`, {
          credentials: "include",
        });
        if (res.ok) {
          const data = await res.json();
          setUser(data);
        }
      } catch (err) {
        console.error("Error fetching user info:", err);
      }
    };
    fetchUser();
  }, []);

  //Handle logout
  const handleLogout = async () => {
    try {
      await fetch(`${import.meta.env.VITE_API_URL}/auth/logout`, {
        method: "POST",
        credentials: "include",
      });

      if (socket.connected) {
        socket.disconnect();
      }

      navigate("/login", { replace: true });
    } catch (err) {
      console.error("Logout failed", err);
    }
  };

  return (
    <Sidebar collapsible="icon" className="group">
      <SidebarHeader className="px-3 py-2">
        <div className="flex items-center gap-2">
          <img src="/logo.svg" alt="EcoSense" className="h-8 w-8" />
          <span className="text-lg font-bold group-data-[collapsible=icon]:hidden">
            <span className="text-[#6FA672]">Eco</span>Sense
            <span className="ml-1 text-muted-foreground">Admin</span>
          </span>
        </div>
      </SidebarHeader>

      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel className="group-data-[collapsible=icon]:hidden">
            Management
          </SidebarGroupLabel>

          <ScrollArea className="h-[calc(100vh-12rem)]">
            <SidebarMenu>
              {NAV.map((item) => {
                const Icon = item.icon;
                const active = item.match(pathname);

                return (
                  <SidebarMenuItem key={item.to} className="mb-1">
                    <SidebarMenuButton
                      asChild
                      isActive={active}
                      tooltip={item.label}
                      className={`
                        flex items-center gap-3 rounded-md transition-all duration-200
                        ${active
                          ? "bg-[#eaf5eb] text-[#4c8050] scale-[1.02]"
                          : "hover:bg-[#eaf5eb] hover:text-[#4c8050] hover:scale-[1.02]"
                        }
                      `}
                    >
                      <Link
                        to={item.to}
                        className="flex items-center gap-3 w-full transition-all duration-200"
                      >
                        <Icon className="h-5 w-5 transition-transform duration-200 group-hover:scale-110" />
                        <span className="text-sm font-medium group-data-[collapsible=icon]:hidden">
                          {item.label}
                        </span>
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                );
              })}
            </SidebarMenu>
          </ScrollArea>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter className="px-3 pb-3">
        <div
          className="
            flex items-center justify-between rounded-md border px-3 py-2
            transition-all duration-300
            group-data-[collapsible=icon]:flex-col group-data-[collapsible=icon]:border-none group-data-[collapsible=icon]:p-0 group-data-[collapsible=icon]:gap-3
          "
        >
          {/* Avatar + user info */}
          <div className="flex items-center gap-2 overflow-hidden group-data-[collapsible=icon]:flex-col group-data-[collapsible=icon]:gap-0">
            <Avatar className="h-9 w-9 shrink-0">
              <AvatarFallback>
                {user?.name ? user.name.slice(0, 2).toUpperCase() : "AD"}
              </AvatarFallback>
            </Avatar>
            <div className="min-w-0 max-w-[130px] group-data-[collapsible=icon]:hidden">
              <p className="truncate text-sm font-medium">
                {user?.name || "Admin User"}
              </p>
              <p className="truncate text-xs text-muted-foreground">
                {user?.email || "admin@example.com"}
              </p>
            </div>
          </div>

          {/* Settings + Logout buttons */}
          <div className="flex items-center gap-2 group-data-[collapsible=icon]:flex-col group-data-[collapsible=icon]:gap-3">
            <button
              onClick={handleLogout}
              aria-label="Logout"
              className="inline-flex p-1 hover:scale-110 hover:text-red-500 transition-transform cursor-pointer"
            >
              <LogOut className="h-4 w-4" />
            </button>
          </div>
        </div>
      </SidebarFooter>

      <SidebarRail />
    </Sidebar>
  );
}