from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math

OUTDIR = Path('output/figures')
OUTDIR.mkdir(parents=True, exist_ok=True)
PNG = OUTDIR / 'improved_rotated_detector_architecture.png'
SVG = OUTDIR / 'improved_rotated_detector_architecture.svg'

W, H = 1900, 1150
img = Image.new('RGB', (W, H), 'white')
draw = ImageDraw.Draw(img)
FONT = r'C:\Windows\Fonts\arial.ttf'
FONT_B = r'C:\Windows\Fonts\arialbd.ttf'

def font(size, bold=False):
    path = FONT_B if bold else FONT
    return ImageFont.truetype(path, size)

C = {
    'text': '#1f2633',
    'subtext': '#5c6575',
    'outline': '#cfd7e6',
    'main': '#fafbff',
    'backbone': '#eef3ff',
    'neck': '#f4f0ff',
    'head': '#fff4ee',
    'conv': '#ffd9ba',
    'module': '#dce8ff',
    'c2f': '#e8c6f2',
    'concat': '#f5cbd3',
    'upsample': '#cceef1',
    'fcm': '#f8efb6',
    'mkp': '#f4d5b9',
    'sppf': '#d8f1f1',
    'headbox': '#dcecb9',
    'conv2d': '#ebe6d8',
    'zoom1': '#fff6d9',
    'zoom2': '#ffeadd',
    'accent': '#6a5acd',
    'orange': '#d9842b',
    'arrow': '#3c4557',
}

svg = []
svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
svg.append('''<defs>
<marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto" markerUnits="strokeWidth">
  <path d="M 0 0 L 10 5 L 0 10 z" fill="#3c4557" />
</marker>
</defs>''')

def sx(v):
    return f'{v:.1f}'

def text(x, y, content, size=24, bold=False, fill=C['text'], anchor='la'):
    f = font(size, bold=bold)
    bbox = draw.textbbox((0, 0), content, font=f)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx, ty = x, y
    if anchor == 'ma':
        tx = x - tw / 2
        ty = y - th / 2
    elif anchor == 'ra':
        tx = x - tw
    draw.text((tx, ty), content, fill=fill, font=f)
    svg_anchor = {'la': 'start', 'ma': 'middle', 'ra': 'end'}[anchor]
    svg.append(f'<text x="{sx(x)}" y="{sx(y + size*0.8)}" text-anchor="{svg_anchor}" font-family="Arial" font-size="{size}" fill="{fill}" font-weight="{"700" if bold else "400"}">{content}</text>')

def round_box(x1, y1, x2, y2, fill, outline=C['outline'], width=2, radius=20):
    draw.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill=fill, outline=outline, width=width)
    svg.append(f'<rect x="{sx(x1)}" y="{sx(y1)}" width="{sx(x2-x1)}" height="{sx(y2-y1)}" rx="{sx(radius)}" ry="{sx(radius)}" fill="{fill}" stroke="{outline}" stroke-width="{width}" />')

def box_label(x1, y1, x2, y2, label, fill, size=22, radius=12, outline='none', width=0, bold=False):
    if outline == 'none':
        round_box(x1, y1, x2, y2, fill, fill, 0, radius)
    else:
        round_box(x1, y1, x2, y2, fill, outline, width, radius)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    lines = label.split('\n')
    f = font(size, bold=bold)
    line_h = size + 4
    total_h = line_h * len(lines) - 4
    for i, line_txt in enumerate(lines):
        bbox = draw.textbbox((0, 0), line_txt, font=f)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = cx - tw / 2
        ty = cy - total_h / 2 + i * line_h - th / 2 + line_h / 2
        draw.text((tx, ty), line_txt, fill=C['text'], font=f)
    svg.append(f'<text x="{sx(cx)}" y="{sx(cy - (len(lines)-1)*(size+4)/2 + size/3)}" text-anchor="middle" font-family="Arial" font-size="{size}" fill="{C["text"]}" font-weight="{"700" if bold else "400"}">')
    for i, line_txt in enumerate(lines):
        dy = '0' if i == 0 else str(size + 4)
        svg.append(f'<tspan x="{sx(cx)}" dy="{dy}">{line_txt}</tspan>')
    svg.append('</text>')

def line(x1, y1, x2, y2, fill=C['arrow'], width=3, arrow_head=True):
    draw.line([x1, y1, x2, y2], fill=fill, width=width)
    if arrow_head:
        ang = math.atan2(y2-y1, x2-x1)
        ah = 10
        aw = 6
        p1 = (x2, y2)
        p2 = (x2 - ah*math.cos(ang) + aw*math.sin(ang), y2 - ah*math.sin(ang) - aw*math.cos(ang))
        p3 = (x2 - ah*math.cos(ang) - aw*math.sin(ang), y2 - ah*math.sin(ang) + aw*math.cos(ang))
        draw.polygon([p1, p2, p3], fill=fill)
        svg.append(f'<line x1="{sx(x1)}" y1="{sx(y1)}" x2="{sx(x2)}" y2="{sx(y2)}" stroke="{fill}" stroke-width="{width}" marker-end="url(#arrow)" />')
    else:
        svg.append(f'<line x1="{sx(x1)}" y1="{sx(y1)}" x2="{sx(x2)}" y2="{sx(y2)}" stroke="{fill}" stroke-width="{width}" />')

def dashed_line(x1, y1, x2, y2, fill=C['accent'], width=3):
    draw.line([x1, y1, x2, y2], fill=fill, width=width)
    svg.append(f'<line x1="{sx(x1)}" y1="{sx(y1)}" x2="{sx(x2)}" y2="{sx(y2)}" stroke="{fill}" stroke-width="{width}" stroke-dasharray="10 8" />')

def circle(x, y, r, fill='white', outline='#62708b', width=2):
    draw.ellipse([x-r, y-r, x+r, y+r], fill=fill, outline=outline, width=width)
    svg.append(f'<circle cx="{sx(x)}" cy="{sx(y)}" r="{sx(r)}" fill="{fill}" stroke="{outline}" stroke-width="{width}" />')

text(950, 18, 'Improved Rotated Vehicle Detection Network', 40, True, anchor='ma')
text(950, 58, 'Backbone - Neck - Rotated Head with Feature Complementary Mapping and Multi-Kernel Perception', 21, False, C['subtext'], 'ma')
box_label(85, 88, 170, 120, 'FCM', C['fcm'], size=18)
text(185, 89, 'proposed feature enhancement', 18)
box_label(420, 88, 505, 120, 'MKP', C['mkp'], size=18)
text(520, 89, 'proposed multi-kernel unit', 18)
box_label(760, 88, 860, 120, 'Conv/CBS', C['conv'], size=18)
text(875, 89, 'basic convolution block', 18)

round_box(20, 140, 1070, 1030, C['main'], '#dde3ef', 3, 28)
round_box(55, 190, 250, 950, C['backbone'], '#cfd8eb', 3, 26)
round_box(310, 240, 705, 760, C['neck'], '#d8d5ec', 3, 26)
round_box(760, 240, 1030, 760, C['head'], '#e7d2c5', 3, 26)
text(152, 200, 'Backbone', 28, True, anchor='ma')
text(507, 250, 'Neck', 28, True, anchor='ma')
text(895, 250, 'Rotated Head', 28, True, anchor='ma')

for i, col in enumerate(['#539bf5', '#57d3ca', '#ff5c5c']):
    draw.rectangle([90 + i*12, 960 - i*12, 150 + i*12, 1020 - i*12], fill=col)
    svg.append(f'<rect x="{90 + i*12}" y="{960 - i*12}" width="60" height="60" fill="{col}" />')
text(150, 1032, 'Input Image', 20, anchor='ma')
line(152, 955, 152, 950)

mods = [
    ('CBS', C['conv']), ('CBS', C['conv']), ('FCM', C['fcm']), ('Conv', C['module']),
    ('FCM', C['fcm']), ('Conv', C['module']), ('FCM', C['fcm']), ('MKP', C['mkp']),
    ('FCM', C['fcm']), ('Conv', C['module']), ('SPPF', C['sppf'])
]
base_x1, base_x2 = 105, 205
y = 875
for i, (lab, col) in enumerate(mods):
    box_label(base_x1, y, base_x2, y+44, lab, col, size=21)
    text(90, y+6, str(i), 18, anchor='ra')
    if i < len(mods)-1:
        line(155, y, 155, y-20)
    y -= 64

left = [(350, 680, 'C2f', C['c2f'], '11'), (350, 605, 'Concat', C['concat'], '12'), (350, 530, 'UP', C['upsample'], '13'),
        (350, 420, 'C2f', C['c2f'], '14'), (350, 345, 'Concat', C['concat'], '15'), (350, 270, 'UP', C['upsample'], '16')]
right = [(585, 680, 'C2f', C['c2f'], '22'), (585, 605, 'Concat', C['concat'], '21'), (585, 530, 'Conv', C['conv'], '20'),
         (585, 420, 'C2f', C['c2f'], '19'), (585, 345, 'Concat', C['concat'], '18'), (585, 270, 'Conv', C['conv'], '17')]
for x, y, lab, col, n in left + right:
    box_label(x, y, x+110, y+44, lab, col, size=21)
    text(x-16, y+6, n, 18, anchor='ra')

line(405, 649, 405, 605)
line(405, 574, 405, 530)
line(405, 464, 405, 420)
line(405, 389, 405, 345)
line(640, 314, 640, 345)
line(640, 389, 640, 420)
line(640, 499, 640, 530)
line(640, 574, 640, 605)
line(640, 649, 640, 680)
line(205, 642, 350, 627)
line(205, 508, 350, 367)
line(205, 374, 350, 697)
line(205, 244, 585, 702)
line(460, 702, 585, 702)
line(460, 442, 585, 367)
line(460, 292, 760, 292)
line(695, 442, 760, 442)
line(695, 702, 760, 702)
line(205, 180, 350, 292)

rows = [(650, 'P3'), (455, 'P4'), (285, 'P5')]
for y, scale in rows:
    text(785, y-25, scale, 18, False, C['subtext'])
    for idx, (lab, col) in enumerate([('CBS', C['conv']), ('DSC', C['headbox']), ('CBS', C['conv'])]):
        yy = y - idx*55
        box_label(790, yy, 875, yy+40, lab, col, size=20)
        box_label(900, yy, 985, yy+40, lab, col, size=20)
        box_label(1000, yy, 1088, yy+40, 'Conv2d', C['conv2d'], size=18)
        line(760, yy+20, 790, yy+20)
        line(875, yy+20, 900, yy+20)
        line(985, yy+20, 1000, yy+20)
    for yy, label in [(y+10, 'Cls'), (y-45, 'Obj'), (y-100, 'RBox / Ori')]:
        text(1100, yy-6, label, 19)

round_box(1090, 140, 1780, 590, C['zoom1'], '#ead69c', 3, 26)
round_box(1090, 640, 1780, 1000, C['zoom2'], '#e9c7ab', 3, 26)
text(1435, 150, 'Feature Complementary Mapping Module (FCM)', 31, True, anchor='ma')
text(1435, 650, 'Multi-Kernel Perception Unit (MKP)', 31, True, anchor='ma')

box_label(1170, 320, 1285, 382, 'Input\nFeatures', '#ffd6b8', size=22)
text(1210, 396, 'Split', 21)
line(1285, 350, 1345, 260)
line(1285, 350, 1345, 455)
box_label(1345, 220, 1485, 282, 'Position\nEmbedding', '#f7f7cf', size=22, outline='#cbbf91', width=2)
box_label(1345, 425, 1485, 487, 'Semantic\nEnrichment', '#dff1cf', size=22, outline='#a7c58b', width=2)
box_label(1515, 305, 1660, 355, 'Channel Guidance', '#ffddb9', size=20)
box_label(1675, 305, 1765, 355, 'Spatial\nAggregation', '#ffe4c7', size=18)
box_label(1505, 435, 1675, 485, 'Complementary Mapping', '#f1c488', size=19)
line(1485, 250, 1515, 330)
line(1485, 456, 1515, 330)
line(1660, 330, 1675, 330)
line(1588, 355, 1588, 435)
line(1675, 460, 1728, 460)
circle(1725, 330, 16)
text(1725, 316, '+', 24, True, '#62708b', 'ma')
line(1741, 330, 1770, 330)
for i in range(4):
    round_box(1778 + i*14, 287 + i*4, 1796 + i*14, 373 + i*4, C['orange'], C['orange'], 1, 4)
text(1814, 392, 'Output Features', 21, anchor='ra')
text(1590, 195, 'S: Sigmoid', 21)
text(1590, 228, 'AvgPool: Adaptive average pooling', 19)
text(1590, 258, 'PWConv: Pointwise convolution', 19)
text(1500, 235, 'w1', 22)
text(1596, 412, 'w2', 22)
circle(1512, 286, 14, fill='#ffd2df', outline='#ffd2df', width=1)
text(1512, 273, 'S', 18, False, '#8a4760', 'ma')
circle(1676, 438, 14, fill='#ffd2df', outline='#ffd2df', width=1)
text(1676, 425, 'S', 18, False, '#8a4760', 'ma')

text(1175, 810, 'X', 34, True)
mkp_x = [1245, 1350, 1455, 1560, 1665, 1770]
mkp_labels = ['DWConv', 'PWConv', 'DWConv', 'PWConv', 'DWConv', 'PWConv']
for x, lab in zip(mkp_x, mkp_labels):
    box_label(x, 760, x+72, 900, lab, '#dfe8fb', size=23, outline='#9fb2d8', width=2)
line(1200, 830, 1245, 830)
for x in mkp_x[:-1]:
    line(x+72, 830, x+105, 830)
line(mkp_x[-1]+72, 830, 1832, 830)
text(1840, 810, "X'", 34, True)
for x, label, col in [(1295, 'k=3', '#87aefb'), (1505, 'k=5', '#6cc7d6'), (1715, 'k=7', '#ffb86f')]:
    round_box(x-34, 928, x+34, 992, col, col, 1, 8)
    text(x, 995, label, 19, anchor='ma')

text(205, 560, 'FCM plug-ins', 18, False, C['accent'])
text(205, 485, 'MKP insertion', 18, False, C['accent'])
dashed_line(205, 560, 1090, 260)
dashed_line(205, 495, 1090, 805)

text(950, 1088, 'Draft for paper revision. Recommended final usage: one main architecture figure plus detachable FCM/MKP subfigures.', 18, False, C['subtext'], 'ma')

img.save(PNG)
svg.append('</svg>')
SVG.write_text('\n'.join(svg), encoding='utf-8')
print(PNG)
print(SVG)
