from export import Exporter
try:
    from exportutils.config import Config as ExportConfig
except ImportError:
    from export import ExportConfig
import gui3d
import mh
import gui
import random
import time


class Mhx2Config(ExportConfig):
    def __init__(self):
        ExportConfig.__init__(self)
        self.useBinary     = False


class DataGeneratorTaskView(gui3d.TaskView):
    def __init__(self, category):
        gui3d.TaskView.__init__(self, category, 'Data generation')
        self.human = gui3d.app.selectedHuman

        box = self.addLeftWidget(gui.GroupBox('Data generation'))
        self.button = box.addWidget(gui.Button('Generate'))
        self.path_label = box.addWidget(gui.TextView('Path'))
        self.path = box.addWidget(gui.TextEdit())
        self.num_label = box.addWidget(gui.TextView('# Meshes'))
        self.num = box.addWidget(gui.TextEdit())
        @self.button.mhEvent
        def onClicked(event):
            num_gen = int(self.num.getText())
            t0 = time.time()

            for i in range(num_gen):
                randomize(self.human, 0.5, True, True, False, True)
                cfg = Mhx2Config()
                cfg.useTPose = False
                cfg.useBinary = True
                cfg.useExpressions = False
                cfg.usePoses = True
                cfg.feetOnGround = True
                cfg.scale, cfg.unit = 0.1, 'meter'
                cfg.setHuman(self.human)
                from . import mh2mhx2
                mh2mhx2.exportMhx2('%s/%s.mhx2' % (self.path.getText(), i), cfg)

                eta = int((num_gen-1-i)*((time.time()-t0)/(i+1))/60)
                gui3d.app.statusPersist('ETA: %s minutes' % eta)


class RandomizeAction(gui3d.Action):
    def __init__(self, human, before, after):
        super(RandomizeAction, self).__init__("Randomize")
        self.human = human
        self.before = before
        self.after = after

    def do(self):
        self._assignModifierValues(self.after)
        return True

    def undo(self):
        self._assignModifierValues(self.before)
        return True

    def _assignModifierValues(self, valuesDict):
        _tmp = self.human.symmetryModeEnabled
        self.human.symmetryModeEnabled = False
        for mName, val in valuesDict.items():
            try:
                self.human.getModifier(mName).setValue(val)
            except:
                pass
        self.human.applyAllTargets()
        self.human.symmetryModeEnabled = _tmp


def randomize(human, symmetry, macro, height, face, body):
    modifierGroups = []
    if macro:
        modifierGroups = modifierGroups + ['macrodetails', 'macrodetails-universal', 'macrodetails-proportions']
    if height:
        modifierGroups = modifierGroups + ['macrodetails-height']
    if face:
        modifierGroups = modifierGroups + ['eyebrows', 'eyes', 'chin', 
                         'forehead', 'head', 'mouth', 'nose', 'neck', 'ears',
                         'cheek']
    if body:
        modifierGroups = modifierGroups + ['pelvis', 'hip', 'armslegs', 'stomach', 'breast', 'buttocks', 'torso']

    modifiers = []
    for mGroup in modifierGroups:
        modifiers = modifiers + human.getModifiersByGroup(mGroup)
    # Make sure not all modifiers are always set in the same order 
    # (makes it easy to vary dependent modifiers like ethnics)
    random.shuffle(modifiers)

    randomValues = {}
    for m in modifiers:
        if m.fullName not in randomValues:
            randomValue = None
            if m.groupName == 'head':
                sigma = 0.1
            elif m.fullName in ["forehead/forehead-nubian-less|more", "forehead/forehead-scale-vert-less|more"]:
                sigma = 0.02
                # TODO add further restrictions on gender-dependent targets like pregnant and breast
            elif "trans-horiz" in m.fullName or m.fullName == "hip/hip-trans-in|out":
                if symmetry == 1:
                    randomValue = m.getDefaultValue()
                else:
                    mMin = m.getMin()
                    mMax = m.getMax()
                    w = float(abs(mMax - mMin) * (1 - symmetry))
                    mMin = max(mMin, m.getDefaultValue() - w/2)
                    mMax = min(mMax, m.getDefaultValue() + w/2)
                    randomValue = getRandomValue(mMin, mMax, m.getDefaultValue(), 0.1)
            elif m.groupName in ["forehead", "eyebrows", "neck", "eyes", "nose", "ears", "chin", "cheek", "mouth"]:
                sigma = 0.1
            elif m.groupName.startswith('macrodetails'):
                if m.fullName == 'macrodetails/Age':
                    randomValue = random.uniform(0.25, 0.75)
                elif m.fullName == 'macrodetails-universal/Weight':
                    randomValue = random.uniform(0.5, 1.0)
                elif m.fullName == 'macrodetails-universal/Muscle':
                    randomValue = random.uniform(0, 0.5)
                else:
                    randomValue = random.random()
            #elif m.groupName == "armslegs":
            #    sigma = 0.1
            else:
                #sigma = 0.2
                sigma = 0.1

            if randomValue is None:
                randomValue = getRandomValue(m.getMin(), m.getMax(), m.getDefaultValue(), sigma)   # TODO also allow it to continue from current value?
            randomValues[m.fullName] = randomValue
            symm = m.getSymmetricOpposite()
            if symm and symm not in randomValues:
                if symmetry == 1:
                    randomValues[symm] = randomValue
                else:
                    m2 = human.getModifier(symm)
                    symmDeviation = float((1-symmetry) * abs(m2.getMax() - m2.getMin()))/2
                    symMin =  max(m2.getMin(), min(randomValue - (symmDeviation), m2.getMax()))
                    symMax =  max(m2.getMin(), min(randomValue + (symmDeviation), m2.getMax()))
                    randomValues[symm] = getRandomValue(symMin, symMax, randomValue, sigma)

    if randomValues.get("macrodetails/Gender", 0) > 0.5 or \
       randomValues.get("macrodetails/Age", 0.5) < 0.2 or \
       randomValues.get("macrodetails/Age", 0.7) < 0.75:
        # No pregnancy for male, too young or too old subjects
        if "stomach/stomach-pregnant-decr|incr" in randomValues:
            randomValues["stomach/stomach-pregnant-decr|incr"] = 0

    oldValues = dict( [(m.fullName, m.getValue()) for m in modifiers] )

    gui3d.app.do( RandomizeAction(human, oldValues, randomValues) )


def getRandomValue(minValue, maxValue, middleValue, sigmaFactor = 0.2):
    rangeWidth = float(abs(maxValue - minValue))
    sigma = sigmaFactor * rangeWidth
    randomVal = random.gauss(middleValue, sigma)
    if randomVal < minValue:
        randomVal = minValue + abs(randomVal - minValue)
    elif randomVal > maxValue:
        randomVal = maxValue - abs(randomVal - maxValue)
    return max(minValue, min(randomVal, maxValue))


category = None
taskview = None


def load(app):
    category = app.getCategory('Data generator')
    taskview = category.addTask(DataGeneratorTaskView(category))


def unload(app):
    pass
