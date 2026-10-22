def obj(properties):
    return {'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}

evidence=obj({'source_uid':{'type':'string'},'quote':{'type':'string','minLength':8,'maxLength':500}})
claim=obj({'text':{'type':'string','minLength':1,'maxLength':800},'evidence':{'type':'array','items':evidence,'minItems':1,'maxItems':4}})
knowledge=obj({**claim['properties'],'kind':{'enum':['convention','precedent','pitfall','contract','rationale']}})
ENGINEERING_SCHEMA=obj({'task_summary':{'anyOf':[claim,{'type':'null'}]},
                       'diagnosis':{'type':'array','items':claim,'maxItems':3},
                       'resolution':{'type':'array','items':claim,'maxItems':3},
                       'knowledge':{'type':'array','items':knowledge,'maxItems':5}})
