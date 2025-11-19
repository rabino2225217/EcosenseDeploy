import { io, Socket } from "socket.io-client";

const API_URL = import.meta.env.VITE_API_URL; 

const socket: Socket = io(API_URL, { 
  withCredentials: true,
  transports: ["websocket"], 
  autoConnect: false
});

export default socket;