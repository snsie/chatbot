from time import sleep
from py_trees.behaviour import Behaviour # used to create Action and Condition nodes (execution nodes)
from py_trees.common import Status # used to create the status of a node (success, failure, running)
from py_trees.composites import Sequence, Selector # used to create a sequence of nodes (parent nodes)
from py_trees.decorators import EternalGuard
from py_trees import logging as log_tree # used for terminal prints (visualization of ticks)

####### pytrees classes ########
class Action(Behaviour):
    def __init__(self, name): # 1-time initialization when node object is created
        super(Action, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))


    def setup(self): # delayed 1-timeinitialization used for initializing members you don't want to in __init__
        self.logger.debug(f"Action::setup {self.name}")

    def initialise(self): # initialization when node is ticked for the first time and anytime the status is not RUNNING thereafter, resets behavior before running
        self.logger.debug(f"Action::initialise {self.name}")

    def update(self): # main function that is called when the node is ticked
        self.logger.debug(f"Action::update {self.name}")
        return Status.SUCCESS
    
    def terminate(self, new_status): # code called when node switches to a non-RUNNING state (SUCCESS or FAILURE)
        self.logger.debug(f"Action::terminate {self.name} to {new_status}")


class Condition(Behaviour):
    def __init__(self, name):
        super(Condition, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

    def setup(self):
        self.logger.debug(f"Condition::setup {self.name}")

    def initialise(self):
        self.logger.debug(f"Condition::initialise {self.name}")

    def update(self):
        self.logger.debug(f"Condition::update {self.name}")
        return Status.FAILURE
    
    def terminate(self, new_status):
        self.logger.debug(f"Condition::terminate {self.name} to {new_status}")


def make_bt():
    root_1 = Selector(name="Selector", memory=False) # figure out which memory to use

    audio_detected_2 = Condition("Audio Detected?")
    process_audio_3 = Sequence("Processing Audio", memory=True)

    root_1.add_children([audio_detected_2, process_audio_3])

    voice_id_on_4 = Condition("Voice ID On?")
    voice_recognized_5 = Selector("Voice Recognized?", memory=False)
    transcribe_audio_6 = Action("Transcribe Audio")
    cora_response_7 = Selector("Cora Response", memory=False)

    process_audio_3.add_children([voice_id_on_4, voice_recognized_5, transcribe_audio_6, cora_response_7])

    known_voice_8 = Condition("Known Voice?")
    unknown_voice_9 = Selector("Unknown Voice", memory=False)
    
    voice_recognized_5.add_children([known_voice_8, unknown_voice_9])

    decline_enrollment_12 = Condition("Decline Enrollment?")
    accept_enrollment_13 = Sequence("Accept Enrollment", memory=True)

    unknown_voice_9.add_children([decline_enrollment_12, accept_enrollment_13])

    get_name_14 = Action("Get Name")
    record_samples_15 = Action("Record Samples")
    embed_voice_16 = Sequence("Embed Voice", memory=True)
    save_db_17 = Action("Save DB")

    accept_enrollment_13.add_children([get_name_14, record_samples_15, embed_voice_16, save_db_17])

    build_vectors_18 = Action("Build Vectors")
    embedded_file_19 = Condition("Embedded File Exists?")

    embed_voice_16.add_children([build_vectors_18, embedded_file_19])

    style_change_10 = Sequence("Style Change", memory=True)
    normal_response_11 = Action("Normal Response")

    cora_response_7.add_children([style_change_10, normal_response_11])

    style_change_command_20 =  Condition("Style Change Command?")
    style_adjustment_21 = Action("Style Adjustment")

    style_change_10.add_children([style_change_command_20, style_adjustment_21])

    return root_1

def main():
    log_tree.level = log_tree.Level.DEBUG
    tree = make_bt()
    for i in range(3):
        print(f"\n----- Tick {i} -----")
        tree.tick_once()
        sleep(1)

if __name__ == "__main__":
    main()