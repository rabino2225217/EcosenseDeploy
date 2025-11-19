import React, { useState } from "react";
import { Link } from "react-router-dom";
import { Check, X, CircleAlert, CheckCircle2, Loader } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "../../components/ui/alert";
import { motion, AnimatePresence } from "framer-motion";

const API_URL = import.meta.env.VITE_API_URL;

export default function Register() {
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [pw, setPw] = useState("");
  const [confirm, setConfirm] = useState("");

  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const passwordChecks = {
    length: pw.length >= 8 && pw.length <= 20,
    upper: /[A-Z]/.test(pw),
    lower: /[a-z]/.test(pw),
    number: /\d/.test(pw),
    special: /[\W_]/.test(pw),
  };

  React.useEffect(() => {
    document.title = "Register | EcoSense";
  }, []);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setMessage(null);

    if (pw !== confirm) {
      setError("Passwords do not match.");
      setLoading(false);
      return;
    }

    try {
      const response = await fetch(`${API_URL}/auth/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ name: username, email, password: pw }),
      });

      if (response.ok) {
        const text = await response.text();
        setMessage(text);
        setUsername("");
        setEmail("");
        setPw("");
        setConfirm("");
      } else {
        const errorText = await response.text();
        setError(errorText || "Registration failed.");
      }
    } catch (err) {
      console.error("Register error:", err);
      setError("Something went wrong. Please try again.");
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
          An AI-Powered System for Detecting Trees and Classifying Croplands
          from Drone Imagery for the{" "}
          <b>DENR - Cordillera Administrative Region</b>
        </p>
      </section>

      {/* RIGHT: form */}
      <section className="relative min-h-screen flex items-center justify-center text-slate-800 p-8 bg-slate-50">
        <div className="w-full max-w-md">
          <div className="flex flex-col items-center mb-6">
            <h1 className="text-5xl font-extrabold tracking-tight text-center">
              <span className="text-[#6FA672]">Eco</span>Sense
            </h1>
          </div>

          <div className="mb-6">
            <h2 className="text-3xl font-bold text-left">Sign Up</h2>
            <p className="text-slate-600 mt-1 font-light text-left">
              Create new account
            </p>
          </div>

          <form onSubmit={onSubmit} className="space-y-4" autoComplete="off">
            {/* Username + Email */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="flex items-center gap-3 rounded-xl bg-white text-slate-900 border border-slate-300 px-4">
                <UserIcon className="h-5 w-5 text-slate-500" />
                <input
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  placeholder="Full Name"
                  className="w-full bg-transparent py-3 outline-none"
                  required
                />
              </div>

              <div className="flex items-center gap-3 rounded-xl bg-white text-slate-900 border border-slate-300 px-4">
                <MailIcon className="h-5 w-5 text-slate-500" />
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="Email"
                  className="w-full bg-transparent py-3 outline-none"
                  required
                />
              </div>
            </div>

            {/* Password */}
            <div className="flex items-center gap-3 rounded-xl bg-white text-slate-900 border border-slate-300 px-4">
              <LockIcon className="h-5 w-5 text-slate-500" />
              <input
                type="password"
                value={pw}
                onChange={(e) => setPw(e.target.value)}
                placeholder="Password"
                className="w-full bg-transparent py-3 outline-none"
                required
              />
            </div>

            {/* Password Requirements */}
            <div className="mt-2 text-xs text-slate-950">
              <ul className="space-y-1">
                <PasswordRequirement
                  valid={passwordChecks.length}
                  text="At least 8 characters"
                />
                <PasswordRequirement
                  valid={passwordChecks.upper}
                  text="An uppercase letter"
                />
                <PasswordRequirement
                  valid={passwordChecks.lower}
                  text="A lowercase letter"
                />
                <PasswordRequirement
                  valid={passwordChecks.number}
                  text="A number"
                />
                <PasswordRequirement
                  valid={passwordChecks.special}
                  text="A special character"
                />
              </ul>
            </div>

            {/* Confirm Password */}
            <div
              className={`flex items-center gap-3 rounded-xl bg-white text-slate-900 border px-4 ${
                error ? "border-red-400" : "border-slate-300"
              }`}
            >
              <LockIcon className="h-5 w-5 text-slate-500" />
              <input
                type="password"
                value={confirm}
                onChange={(e) => {
                  setConfirm(e.target.value);
                  if (pw === e.target.value) setError(null);
                }}
                placeholder="Confirm Password"
                className="w-full bg-transparent py-3 outline-none"
                required
              />
            </div>

            {/* Alerts for error/success */}
            <AnimatePresence mode="wait">
              {(error || message) && (
                <motion.div
                  key={error ? "error" : "message"}
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  transition={{ duration: 0.25, ease: "easeInOut" }}
                  className="overflow-hidden"
                >
                  {error && (
                    <Alert
                      variant="destructive"
                      className="bg-red-50 border-red-300 text-red-700 mb-1"
                    >
                      <CircleAlert className="h-4 w-4" />
                      <AlertDescription>{error}</AlertDescription>
                    </Alert>
                  )}

                  {message && (
                    <Alert className="mt-2 bg-emerald-50 border-emerald-300 text-emerald-800">
                      <CheckCircle2 className="h-4 w-4" />
                      <AlertDescription>{message}</AlertDescription>
                    </Alert>
                  )}
                </motion.div>
              )}
            </AnimatePresence>

            <button
              type="submit"
              className="mt-2 w-full rounded-xl bg-[#6FA672] py-3 font-medium text-white hover:bg-emerald-700 focus:ring-2 focus:ring-emerald-300 disabled:opacity-50 flex items-center justify-center cursor-pointer"
            >
              {loading ? (
                <>
                  Registering
                  <Loader className="h-3 w-3 ml-2 animate-spin" />
                </>
              ) : (
                "Register"
              )}
            </button>

            <div className="h-px bg-slate-200 my-2" />

            <p className="text-center text-xs text-slate-600">
              Already have an account?{" "}
              <Link
                to="/login"
                state={{ fromRegister: true }}
                className="text-emerald-600 hover:underline"
              >
                Log In
              </Link>
            </p>
          </form>
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
function MailIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" {...props}>
      <path d="M20 4H4a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V6a2 2 0 0 0-2-2Zm-1.4 4.25-6.25 4.38a1 1 0 0 1-1.7 0L4.4 8.25a1 1 0 1 1 1.2-1.6L12 10l6.4-3.35a1 1 0 1 1 1.2 1.6Z" />
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

function PasswordRequirement({
  valid,
  text,
}: {
  valid: boolean;
  text: string;
}) {
  return (
    <li className="flex items-center gap-2 text-slate-500">
      {valid ? (
        <Check className="w-4 h-4 text-green-500" />
      ) : (
        <X className="w-4 h-4 text-slate-400" />
      )}
      <span>{text}</span>
    </li>
  );
}
