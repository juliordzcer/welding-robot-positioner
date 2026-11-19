#!/usr/bin/env python3
"""Small window to follow the welding task and restart it.

Shows the task time, the positioner angles and the manipulabilities while the
task runs, and the results when it ends. The Restart button calls
/welding/restart.
"""
import json
import threading
import tkinter as tk

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

BG, FG, MUTED, ACCENT = '#2e3440', '#eceff4', '#a3acbd', '#ebcb8b'


class Panel(Node):
    def __init__(self):
        super().__init__('welding_panel')
        self.status = None
        self.create_subscription(String, '/welding/status', self.on_status, 10)
        self.client = self.create_client(Trigger, '/welding/restart')

    def on_status(self, msg):
        self.status = json.loads(msg.data)

    def restart(self):
        if self.client.service_is_ready():
            self.client.call_async(Trigger.Request())


def main():
    rclpy.init()
    node = Panel()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    root = tk.Tk()
    root.title('Welding task')
    root.configure(bg=BG, padx=14, pady=12)
    root.resizable(False, False)
    title = tk.Label(root, text='Welding task', font=('Sans', 13, 'bold'), bg=BG, fg=ACCENT)
    title.pack(anchor='w')
    live = tk.Label(root, text='Waiting for the simulation…', font=('Monospace', 10),
                    justify='left', bg=BG, fg=FG)
    live.pack(anchor='w', pady=(6, 6))
    bar = tk.Canvas(root, width=300, height=8, bg='#4c566a', highlightthickness=0)
    bar.pack(anchor='w')
    fill = bar.create_rectangle(0, 0, 0, 8, fill=ACCENT, width=0)
    result = tk.Label(root, text='', font=('Monospace', 10), justify='left', bg=BG, fg=FG)
    result.pack(anchor='w', pady=(8, 8))
    tk.Button(root, text='Restart', command=node.restart, bg='#5e81ac', fg=FG,
              activebackground='#81a1c1', relief='flat', padx=16, pady=4).pack(anchor='e')

    def refresh():
        s = node.status
        if s:
            live.config(text=(f"Mode:  {s['mode']}    State: {s['state']}\n"
                              f"t = {s['t']:5.1f} / {s['T']:.0f} s\n"
                              f"Tilt φ1 = {s['phi1_deg']:6.1f}°   Rotation φ2 = {s['phi2_deg']:7.1f}°\n"
                              f"wT* = {s['wT']:.3f}   wR = {s['wR']:.3f}   gap = {s['gap_mm']:.0f} mm"
                              + (f"\nSeam: {s['seam']}" if s.get('seam') else '')))
            bar.coords(fill, 0, 0, 300 * s['t'] / s['T'], 8)
            r = s.get('results')
            if r:
                result.config(fg=ACCENT, text=(
                    'Results\n'
                    f"  Objective f (eq. 24): {r['f']}\n"
                    f"  wT*  min {r['wT_min']:.3f}   mean {r['wT_mean']:.3f}\n"
                    f"  wR   min {r['wR_min']:.3f}   mean {r['wR_mean']:.3f}\n"
                    f"  Smallest torch-chuck gap: {r['clearance_min_mm']:.0f} mm"
                    + (f"\n  Cycle time: {r['duration_s']:.1f} s (arc on {r['arc_s']:.1f} s)"
                       if s.get('seam') else '')))
            else:
                result.config(text='')
        root.after(100, refresh)

    refresh()
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
