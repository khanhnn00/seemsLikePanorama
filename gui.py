import functools
from os import path, listdir
from tkinter import *
from tkinter import scrolledtext, filedialog
from tkinter.ttk import *
import panorama

window = Tk()
window.title('Panorama Stitching')
# window.geometry('640x480')

control = Frame(window)
config1 = Frame(window)
config2 = Frame(window)
config3 = Frame(window)
log = Frame(window)

#Widgets for config1
lb_desc = Label(config1, text='Descriptor')
combo_desc = Combobox(config1, justify='center')
combo_desc['value'] = ('ORB')

lb_conf = Label(config1, text='Match confidence')
spin_conf = Spinbox(config1, from_=0, to=1, format="%.1f", increment=0.1, justify='center')

try_cuda = IntVar()
check_accel = Checkbutton(config1, text='GPU acceleration (for NVIDIA)', variable=try_cuda, onvalue=1, offvalue=0)

# Set default value
def desc_handler(event, spin):
    current = combo_desc.current()
    if current == 0:
        spin.delete(0, 'end')
        spin.insert(0, 0.3)
    elif current != -1:
        spin.delete(0, 'end')
        spin.insert(0, 0.65)
desc_handler_wrap = functools.partial(desc_handler, spin=spin_conf)
combo_desc.bind('<<ComboboxSelected>>', desc_handler_wrap)

#Widgets for config2
lb_wave = Label(config2, text='Straightening')
combo_wave = Combobox(config2, justify='center')
combo_wave['value'] = ('Horizontal', 'Vertical', 'No')

lb_warp = Label(config2, text='Warp surface')
combo_warp = Combobox(config2, justify='center')
combo_warp['value'] = ('Spherical', 'Plane', 'Cylindrical')

#Widgets for config3
lb_blend = Label(config3, text='Blending')
combo_blend = Combobox(config3, justify='center')
combo_blend['value'] = ('Multiband')

lb_str = Label(config3, text='strength')
spin_str = Spinbox(config3, from_=0, to=100, justify='center')

# Set default value
def blend_handler(event, spin):
    current = combo_blend.current()
    if current == 2:
        spin.delete(0, 'end')
        spin.insert(0, 0)
        spin.configure(state='disabled')
    elif current != -1:
        spin.configure(state='normal')
        spin.delete(0, 'end')
        spin.insert(0, 5)        
blend_handler_wrap = functools.partial(blend_handler, spin=spin_str)
combo_blend.bind('<<ComboboxSelected>>', blend_handler_wrap)

#Widgets for log
log_txt = scrolledtext.ScrolledText(log, height=20)

#Control button commands
image_list = []
def browse_file(log):
    files = filedialog.askopenfilenames(parent=window, title='Choose images to make panorama')
    files = window.tk.splitlist(files)
    log.insert(INSERT, 'Images obtained: \n')
    for file in files:
        image_list.append(file)
        log.insert(INSERT, file)
        log.insert(INSERT, '\n')
        # print "I got %d bytes from this file." % len(data)
    # print(image_list[-1])


save_name = "F:/work/cs231/project/examination/ijcv2007/results/result.jpg"
# def save_file(log):   
#     save_name.append("F:/work/cs231/project/result")
#     log.insert(INSERT, 'Output image name and save location: '+save_name[-1]+'\n')

def run(desc, conf, accel, wave, warp, blend, strength, log):
    log.insert(INSERT, 'Running...\n')
    pano = panorama.Panorama(save_name, image_list, desc.get(), accel.get(), float(conf.get()), wave.get(), warp.get().lower(), blend.get(), int(strength.get()))
    log.insert(INSERT, '\tReading images\n')
    pano.read_images()
    log.insert(INSERT, pano.log)
    pano.clear_log()
    log.insert(INSERT, '\tMatching\n')
    pano.match()
    log.insert(INSERT, pano.log)
    pano.clear_log()
    log.insert(INSERT, '\tStitching\n')
    pano.stitch()
    log.insert(INSERT, pano.log)
    pano.clear_log()
    log.insert(INSERT, '\tRefining\n')
    pano.refine()
    log.insert(INSERT, pano.log)
    pano.clear_log()
    log.insert(INSERT, '...Done\n')

browse_folder_wrap = functools.partial(browse_file, log=log_txt)
# save_file_wrap = functools.partial(save_file, log=log_txt)
run_wrap = functools.partial(run, desc=combo_desc, conf=spin_conf, accel=try_cuda, wave=combo_wave, warp=combo_warp, blend=combo_blend, strength=spin_str, log=log_txt)

# Packing
lb_desc.pack(side='left')
combo_desc.pack(side='left', padx=10)
lb_conf.pack(side='left', expand=True)
spin_conf.pack(side='left', padx=10)
check_accel.pack(side='right', expand=True)

lb_wave.pack(side='left')
combo_wave.pack(side='left', expand=True, fill='x', padx=10)
lb_warp.pack(side='left', padx=5)
combo_warp.pack(side='left', expand=True, fill='x')

lb_blend.pack(side='left')
combo_blend.pack(side='left', expand=True, fill='x', padx=10)
lb_str.pack(side='left', padx=5)
spin_str.pack(side='left', expand=True, fill='x')

log_txt.pack(expand=True, fill='both')

# Widgets for control
btn_in = Button(control, text='Input', command=browse_folder_wrap).pack(side='left', expand=True, fill='x')
# btn_out = Button(control, text='Output', command=save_file_wrap).pack(side='left', expand=True, fill='x')
btn_run = Button(control, text='Run', command=run_wrap).pack(side='right', expand=True, fill='x')

control.pack(expand=True, fill='x')
config1.pack(expand=True, fill='x', pady=5)
config2.pack(expand=True, fill='x', pady=5)
config3.pack(expand=True, fill='x', pady=5)
log.pack(expand=True, fill='both')

window.mainloop()