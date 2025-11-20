import socket from "../services/socket";
import React, { useState } from "react";
import { Link, useNavigate, useLocation } from "react-router-dom";
import { Alert, AlertDescription } from "../components/ui/alert";
import { CircleAlert, Loader } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

const API_URL = import.meta.env.VITE_API_URL;

export default function Login() {
  const [email, setEmail] = useState("");
  const [pw, setPw] = useState("");
  const [show, setShow] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const navigate = useNavigate();
  const location = useLocation();
  const fromRegister = location.state?.fromRegister || false;
  const [checking, setChecking] = useState(!fromRegister);

  React.useEffect(() => {
    document.title = "Login | EcoSense";
  }, []);

  //Check if there are existing session and then navigate accordingly
  React.useEffect(() => {
    if (fromRegister) return;

    const checkSession = async () => {
      try {
        const res = await fetch(`${API_URL}/auth/me`, {
          credentials: "include",
        });
        if (res.ok) {
          const data = await res.json();
          navigate(data.role === "Admin" ? "/admin" : "/app", {
            replace: true,
          });
          return;
        }
      } catch (err) {
        console.error("Session check failed:", err);
      } finally {
        setChecking(false);
      }
    };

    checkSession();
  }, [navigate, fromRegister]);

  if (checking) {
    return (
      <div className="flex items-center justify-center h-screen text-gray-500 space-x-2">
        <span className="text-sm font-medium">Loading EcoSense</span>
        <Loader className="h-5 w-5 animate-spin text-[#6FA672]" />
      </div>
    );
  }

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setMessage(null);

    //Handle login
    try {
      const response = await fetch(`${API_URL}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ email, password: pw }),
      });

      if (response.ok) {
        const data = await response.json();

        socket.connect();

        navigate(data.redirect, { replace: true });
      } else {
        const errorText = await response.text();
        setMessage(errorText || "Login failed.");
      }
    } catch (err) {
      console.error("Login error:", err);
      setMessage("Something went wrong. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen grid grid-cols-1 md:grid-cols-[60%_40%]">
      {/* LEFT: background */}
      <section className="relative hidden md:block min-h-screen">
        <img
          src="/images/background.svg"
          alt=""
          className="absolute inset-0 h-full w-full object-cover"
        />
        <div className="absolute inset-0 bg-black/30" />
        <div className="relative z-10 p-6 flex items-center gap-3">
          <img src="/denr-logo.svg" alt="DENR-CAR" className="h-25 w-25" />
          <div className="text-white">
            <p className="text-[20px] font-semibold">
              DENR - Cordillera Administrative Region
            </p>
          </div>
        </div>
        <p className="absolute bottom-6 left-6 right-6 z-10 text-center text-white text-sm">
          An AI-Powered System for Detecting Trees and Crop Plots
          from Drone Imagery for the{" "}
          <b>DENR - Cordillera Administrative Region</b>
        </p>
      </section>

      {/* RIGHT: form */}
      <section className="relative min-h-screen flex items-center justify-center text-slate-800 p-8 bg-slate-50">
        <div className="w-full max-w-3xl flex flex-col items-center justify-center text-center">
          <div className="mb-18 flex items-center justify-center">
            <h1 className="text-5xl md:text-6xl font-extrabold tracking-tight">
              <span className="text-[#6FA672]">Eco</span>Sense
            </h1>
          </div>

          <div className="w-full flex justify-center">
            <div className="w-full max-w-md">
              <h2 className="text-3xl font-bold text-left">Welcome!</h2>
              <p className="text-slate-600 mt-1 font-light text-left">
                Please login to your account.
              </p>

              <form
                onSubmit={onSubmit}
                className="mt-6 space-y-4"
                autoComplete="off"
              >
                {/* Email */}
                <label className="block w-full">
                  <span className="sr-only">Email</span>
                  <div className="flex items-center gap-3 rounded-xl bg-white text-slate-900 border border-slate-300 px-4">
                    <UserIcon className="h-5 w-5 text-slate-500" />
                    <input
                      type="email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      placeholder="Email"
                      className="w-full bg-transparent py-3 outline-none"
                      required
                      autoComplete="off"
                    />
                  </div>
                </label>

                {/* Password */}
                <label className="block w-full">
                  <span className="sr-only">Password</span>
                  <div className="flex items-center gap-3 rounded-xl bg-white text-slate-900 border border-slate-300 px-4">
                    <LockIcon className="h-5 w-5 text-slate-500" />
                    <input
                      type={show ? "text" : "password"}
                      value={pw}
                      onChange={(e) => setPw(e.target.value)}
                      placeholder="Password"
                      className="w-full bg-transparent py-3 outline-none"
                      required
                      autoComplete="off"
                    />
                    <button
                      type="button"
                      onClick={() => setShow((s) => !s)}
                      disabled={pw.trim().length === 0}
                      className={`text-sm text-slate-600 hover:text-slate-800 cursor-pointer ${
                        pw.trim().length === 0
                          ? "opacity-50 cursor-not-allowed hover:text-slate-600"
                          : ""
                      }`}
                    >
                      {show ? "Hide" : "Show"}
                    </button>
                  </div>
                </label>

                {/* Alerts for error/success */}
                <AnimatePresence mode="wait">
                  {message && (
                    <motion.div
                      key="login-alert"
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      exit={{ opacity: 0, height: 0 }}
                      transition={{ duration: 0.25, ease: "easeInOut" }}
                      className="overflow-hidden"
                    >
                      <Alert
                        variant="destructive"
                        className="mt-1 bg-red-50 border-red-300 text-red-700 mb-1"
                      >
                        <CircleAlert className="h-4 w-4" />
                        <AlertDescription>{message}</AlertDescription>
                      </Alert>
                    </motion.div>
                  )}
                </AnimatePresence>

                <button
                  type="submit"
                  disabled={loading}
                  className="mt-6 w-full rounded-xl bg-[#6FA672] py-3 font-medium text-white hover:bg-emerald-700 focus:ring-2 focus:ring-emerald-300 disabled:opacity-50 flex items-center justify-center cursor-pointer"
                >
                  {loading ? (
                    <>
                      Logging in
                      <Loader className="h-3 w-3 ml-2 animate-spin" />
                    </>
                  ) : (
                    "Log In"
                  )}
                </button>

                <div className="h-px bg-slate-200 my-2" />

                <p className="text-center text-xs text-slate-600">
                  No account yet?{" "}
                  <Link
                    to="/register"
                    className="text-emerald-600 hover:underline"
                  >
                    Sign Up
                  </Link>
                </p>
              </form>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}

/* icons */
function UserIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" {...props}>
      <path d="M12 12a5 5 0 1 0-5-5 5 5 0 0 0 5 5Zm0 2c-4.42 0-8 2.24-8 5v1h16v-1c0-2.76-3.58-5-8-5z" />
    </svg>
  );
}
function LockIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" {...props}>
      <path d="M17 8h-1V6a4 4 0 0 0-8 0v2Z" />
      <path d="M7 8h10a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2Z" />
    </svg>
  );
}
