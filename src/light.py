class LightController:
    """Safe demo light state. Replace these methods with a smart-device API if needed."""
    def __init__(self):
        self.is_on = False

    def turn_on(self):
        self.is_on = True
        print("[LIGHT] ON")

    def turn_off(self):
        self.is_on = False
        print("[LIGHT] OFF")
