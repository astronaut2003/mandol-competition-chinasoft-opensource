"""App-only prompt/schema guard; no relaxation of the substrate source audit."""
C_RELATIONS={'ADD':'INDEPENDENT','KEEP_BOTH':'INDEPENDENT','NO_CHANGE':'UNCERTAIN',
             'REINFORCE':'EQUIVALENT','UPDATE':'EQUIVALENT','SUPERSEDE':'REPLACEMENT','CLOSE':'RETRACTION'}
C_GUARD='The op/value_relation pair MUST be exactly '+str(C_RELATIONS)+'. If compatible_existing is empty, a RETRACT assertion uses NO_CHANGE, value_relation UNCERTAIN, target_uid null. Do not use RETRACTION with NO_CHANGE. Do not create a positive profile from a retraction.'
A_GUARD='valid_from and valid_to are null unless the source gives an explicit complete calendar DAY in YYYY-MM-DD form. A month such as 2026-11 is NOT a calendar day: keep it verbatim in time_qualifier, set both date fields null. Never invent a day from a month, relative time or upload timestamp. Plans remain plans and goals, not completed events.'
