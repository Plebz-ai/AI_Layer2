class PersonaService:
    def __init__(self):
        self.personas = {}

    def add_persona(self, persona_id, persona_data):
        self.personas[persona_id] = persona_data

    def get_persona(self, persona_id):
        return self.personas.get(persona_id, None)

    def list_personas(self):
        return self.personas