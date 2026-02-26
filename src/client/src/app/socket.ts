import io from "socket.io-client";

const socket = io("ws://localhost:1234");

export default socket;
