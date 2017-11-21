import glob

MHM_PATH = 'D:/passenger_screening_algorithm_challenge/cache/generate_random_models/0/1024'
MHX2_PATH = 'D:/passenger_screening_algorithm_challenge/input/makehuman/generated'

mhx2 = G.app.getPlugin('9_export_mhx2')
cfg = mhx2.Mhx2Config()
cfg.useTPose = False
cfg.useBinary = False
cfg.useExpressions = False
cfg.usePoses = True
cfg.feedOnGround = True
cfg.scale, cfg.unit = 0.1, 'meter'
cfg.setHuman(G.app.selectedHuman)

for file in glob.glob('%s/*.mhm' % MHM_PATH):
    G.app.loadHumanMHM(file)
    file = file.replace('\\', '/')
    out = '%s/%s.mhx2' % (MHX2_PATH, file.split('/')[-1].replace('.mhm', ''))
    mhx2.mh2mhx2.exportMhx2(out, cfg)