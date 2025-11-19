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
import { toast } from "sonner";
import { ScrollArea } from "../ui/scroll-area";
import { Avatar, AvatarFallback } from "../ui/avatar";
import {
  FolderKanban,
  Layers,
  Map,
  FileBarChart,
  LogOut,
  HelpCircle,
} from "lucide-react";

// Dark mode
import { ThemeToggle } from "../ui/toggle";
import { useTheme } from "../../lib/use-darkmode";
// Analysis Status
import { useAnalysisStatus } from "../../routes/client/analysis-status";
import { Button } from "../ui/button";
import { Sparkles } from "lucide-react";
import { useQuickGuide } from "./quick-guide-modal";

const NAV = (selectedProjectId?: string) => [
  {
    label: "All projects",
    to: "/app",
    icon: FolderKanban,
    match: (p: string) => p === "/app" || p === "/app/",
    special: "all",
  },
  ...(selectedProjectId
    ? [
        {
          label: "Land Analysis",
          to: `/app/projects/${selectedProjectId}/analysis`,
          icon: Layers,
          match: (p: string) => p.includes("/analysis"),
        },
        {
          label: "Map View",
          to: `/app/projects/${selectedProjectId}/mapview`,
          icon: Map,
          match: (p: string) => p.includes("/mapview"),
        },
        {
          label: "Summary & Downloads",
          to: `/app/projects/${selectedProjectId}/summary`,
          icon: FileBarChart,
          match: (p: string) => p.includes("/summary"),
        },
      ]
    : []),
];

type User = {
  _id: string;
  name: string;
  email: string;
};

export function AppShell({ children }: { children: React.ReactNode }) {
  const { pathname } = useLocation();
  const navigate = useNavigate();
  const [user, setUser] = React.useState<User | null>(null);
  const [projectName, setProjectName] = React.useState<string | null>(null);
  const projectMatch = pathname.match(/^\/app\/projects\/([^/]+)/);
  const selectedProjectId = projectMatch ? projectMatch[1] : undefined;

  // quick-guide
  const { analyzing } = useAnalysisStatus();
  const { openTour, openInfo } = useQuickGuide();

  //Fetch user info
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
        console.error(err);
      }
    };
    fetchUser();
  }, []);

  //Fetch admin updates on user details
  React.useEffect(() => {
    const handleUserUpdated = (updatedUser: User) => {
      setUser((prev) => {
        if (prev && prev._id === updatedUser._id) {
          return { ...prev, ...updatedUser };
        }
        return prev;
      });
    };

    socket.on("user:updated", handleUserUpdated);
    return () => {
      socket.off("user:updated", handleUserUpdated);
    };
  }, []);

  //Force logout
  React.useEffect(() => {
    socket.on("user:forceLogout", () => {
      toast.warning("Your session has expired. Please log in again.");
      navigate("/login", { replace: true });
    });

    return () => {
      socket.off("user:forceLogout");
    };
  }, []);

  //User deactivated
  React.useEffect(() => {
    const handleUserDeactivated = () => {
      toast.error("Your account has been deactivated by an admin.");
      navigate("/login", { replace: true });
    };

    socket.on("user:deactivated", handleUserDeactivated);

    return () => {
      socket.off("user:deactivated", handleUserDeactivated);
    };
  }, [navigate]);

  //Fetch project name for sidebar
  React.useEffect(() => {
    if (!selectedProjectId) {
      setProjectName(null);
      return;
    }

    const fetchProject = async () => {
      if (analyzing) {
        return;
      }

      try {
        const res = await fetch(
          `${import.meta.env.VITE_API_URL}/project/${selectedProjectId}`,
          {
            credentials: "include",
          }
        );
        const json = await res.json();
        if (res.status === 404) {
          toast.error("Project not found.");
          navigate("/app", { replace: true });
          return;
        }
        setProjectName(json.data?.name || null);
      } catch (err) {
        console.error(err);
      }
    };

    fetchProject();
  }, [selectedProjectId, analyzing, navigate]);

  //User deleted
  React.useEffect(() => {
    const handleUserDeleted = ({ _id }: { _id: string }) => {
      fetch(`${import.meta.env.VITE_API_URL}/auth/me`, {
        credentials: "include",
      })
        .then((res) => {
          if (!res.ok) {
            toast.error("Your account has been deleted by an admin.");
            navigate("/login", { replace: true });
          }
        })
        .catch(() => {
          toast.error("Your account has been deleted by an admin.");
          navigate("/login", { replace: true });
        });
    };

    socket.on("user:deleted", handleUserDeleted);

    return () => {
      socket.off("user:deleted", handleUserDeleted);
    };
  }, [navigate]);

  //Handle project rename and delete
  React.useEffect(() => {
    if (!selectedProjectId) return;

    //Project renamed
    const handleRename = ({ id, name }: { id: string; name: string }) => {
      if (id === selectedProjectId) setProjectName(name);
    };

    //Project deleted
    const handleDelete = ({ id }: { id: string }) => {
      if (id === selectedProjectId) {
        toast.error("Project not found.");
        navigate("/app", { replace: true });
        setProjectName(null);
      }
    };

    socket.on("project:renamed", handleRename);
    socket.on("project:deleted", handleDelete);

    return () => {
      socket.off("project:renamed", handleRename);
      socket.off("project:deleted", handleDelete);
    };
  }, [selectedProjectId, navigate]);

  return (
    <SidebarProvider className="select-none">
      {/* Pass user to AppSidebar */}
      <AppSidebar user={user} />
      <SidebarInset>
        {/* Top bar */}
        <header className="flex h-14 items-center gap-2 border-b px-3">
          {/* left side */}
          <div className="flex items-center gap-2">
            <SidebarTrigger />
            <span className="text-base font-semibold">
              {projectName ?? "Home"}
            </span>
          </div>

          <Button
            type="button"
            onClick={openTour}
            className="
              ml-auto
              hidden md:inline-flex items-center gap-1.5
              rounded-full h-[32px] px-3 text-sm font-medium
              border border-black/70 text-black bg-transparent
              hover:bg-black/5
              dark:border-white/70 dark:text-white dark:hover:bg-white/5
              transition cursor-pointer
            "
          >
            <Sparkles className="h-4 w-4 cursor-pointer" />
            Quick Guide
          </Button>

          {/* info/help icon – static info modal */}
          <button
            onClick={openInfo}
            className="p-0 pr-4 bg-transparent border-0 text-slate-600 hover:text-slate-900 dark:text-slate-200 dark:hover:text-white transition"
            aria-label="Open quick guide info"
          >
            <HelpCircle className="h-5 w-5 cursor-pointer" />
          </button>
        </header>

        {analyzing && (
          <div className="bg-amber-100 dark:bg-amber-900/40 text-amber-900 dark:text-amber-100 text-xs px-4 py-2">
            Image processing is in progress. Some actions are temporarily
            disabled.
          </div>
        )}

        {/* Page content */}
        <div className="flex-1 p-4 md:p-6">{children}</div>
      </SidebarInset>
    </SidebarProvider>
  );
}

export function AppSidebar({ user }: { user: User | null }) {
  const { pathname } = useLocation();
  const navigate = useNavigate();
  const { analyzing } = useAnalysisStatus();
  const { resetToLightMode } = useTheme({ isAdmin: false });
  const projectMatch = pathname.match(/^\/app\/projects\/([^/]+)/);
  const selectedProjectId = projectMatch ? projectMatch[1] : undefined;

  // Handle logout - UPDATED VERSION
  const handleLogout = async () => {
    if (analyzing) {
      toast.warning(
        "An image is currently being processed. Please wait until it finishes."
      );
      return;
    }

    try {
      await fetch(`${import.meta.env.VITE_API_URL}/auth/logout`, {
        method: "POST",
        credentials: "include",
      });

      // Reset theme to light mode before navigating
      resetToLightMode();

      if (socket.connected) {
        socket.disconnect();
      }

      navigate("/login", { state: { loggedOut: true }, replace: true });
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
          </span>
        </div>
      </SidebarHeader>

      <SidebarContent className="flex flex-col">
        <SidebarGroup>
          <SidebarGroupLabel className="group-data-[collapsible=icon]:hidden">
            Menu
          </SidebarGroupLabel>

          <ScrollArea className="flex-1" data-qg="app-sidebar">
            <SidebarMenu>
              {NAV(selectedProjectId).map((item) => {
                const Icon = item.icon;
                const active = item.match(pathname);
                const isAllProjects = item.special === "all";

                return (
                  <SidebarMenuItem key={item.to} className="mb-2">
                    <SidebarMenuButton
                      asChild
                      isActive={active}
                      tooltip={item.label}
                      className={`
                        flex items-center gap-3 rounded-md transition-all duration-200
                        ${
                          isAllProjects
                            ? "h-11 pb-2 border-b border-gray-200 dark:border-gray-700"
                            : "py-3"
                        }
                        text-gray-800 dark:text-gray-200
                        ${
                          active
                            ? "bg-green-100 dark:bg-green-900/40 font-semibold"
                            : "hover:bg-gray-100 dark:hover:bg-gray-800"
                        }
                        ${analyzing ? "opacity-60 cursor-not-allowed" : ""}
                      `}
                    >
                      <Link
                        to={item.to}
                        onClick={(e) => {
                          if (analyzing) {
                            e.preventDefault();
                            toast.warning(
                              "Image processing is in progress. Please wait until it finishes."
                            );
                          }
                        }}
                        className="gap-3 flex items-center w-full transition-all duration-200"
                      >
                        <Icon
                          className={`${
                            isAllProjects ? "h-6 w-6" : "h-5 w-5"
                          } transition-transform duration-200`}
                        />

                        <span
                          className={`group-data-[collapsible=icon]:hidden ${
                            isAllProjects
                              ? "text-base md:text-lg font-semibold"
                              : "text-sm font-medium"
                          }`}
                        >
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
            flex items-center justify-between rounded-md border border-gray-200 dark:border-gray-700
            px-3 py-2 transition-all duration-300
            hover:bg-gray-100 dark:hover:bg-gray-950
            group-data-[collapsible=icon]:flex-col 
            group-data-[collapsible=icon]:border-none 
            group-data-[collapsible=icon]:p-0 
            group-data-[collapsible=icon]:gap-3
          "
        >
          {/* Avatar + user info */}
          <div className="flex items-center gap-2 overflow-hidden group-data-[collapsible=icon]:flex-col group-data-[collapsible=icon]:gap-0">
            <Avatar className="h-9 w-9 shrink-0">
              <AvatarFallback>
                {user?.name
                  ? user.name
                      .split(" ")
                      .map((n) => n[0])
                      .join("")
                      .slice(0, 2)
                      .toUpperCase()
                  : " "}
              </AvatarFallback>
            </Avatar>
            <div className="min-w-0 max-w-[130px] group-data-[collapsible=icon]:hidden">
              <p className="truncate text-sm font-medium">{user?.name}</p>
              <p className="truncate text-xs text-muted-foreground">
                {user?.email}
              </p>
            </div>
          </div>

          {/* Theme toggle + Logout */}
          <div className="flex items-center gap-2 group-data-[collapsible=icon]:flex-col group-data-[collapsible=icon]:gap-3">
            <ThemeToggle />
            <button
              onClick={handleLogout}
              aria-label="Logout"
              disabled={analyzing}
              className={`
                inline-flex p-1 transition-transform
                ${analyzing ? "opacity-60 cursor-not-allowed" : "hover:scale-110 hover:text-red-500 cursor-pointer"}
              `}
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
