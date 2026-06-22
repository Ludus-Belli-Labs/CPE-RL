
# Behaviour of agent:

# detection:
# - detects vignette information (starting time, duration, target)
# - detects vessel position (latitude, longitude)
# - detects vignette target (name, guid, posture)
# - detects target position (latitude, longitude)
# - detects contacts (name, guid, posture)
# - detects contact position (latitude, longitude)

# movement:
# - explore environment (longitude, latitude, radius)
# - move closer to the target (longitude, latitude, target.longitude, target.latitude)
# - move away from enemy (longitude, latitude, name, guid, contact.longitude, contact.latitude, damage) 

# attack:
# - attack enemy (name, guid, contact.longitude, contact.latitude, posture)
# - attack target (name, guid, target.longitude, target.latitude)

# status:
# - damage (name, damage%, fire, flood)
# - radar (on/off)
# - time left (starting time, duration)
# - speed (throttle)
# - waypoint (longitude, latitude, isWater)
# - vignette status (status)


#To Add
# - weather 
# - damage for cenrain parts of the vessle (sensors, weapons, engine)



# Test commands --------------------------------


def GetTitle():
    return f"GetScenarioTitle()"

def GetSelectedUnits():
    return (
        "ScenEdit_SelectedUnits()"
    )
def StartSimulation():
    return f"VP_RunSimulation( )"



# Movement ----------------------------------------


def SetUnitWaypoint(unit_name: str, side: str, lat: float, lon: float) -> str:
    return (
        f"(function() "
        f"local u = ScenEdit_GetUnit({{unitname='{unit_name}', side='{side}'}}) "
        f"if u and (#u.course == 0) then "
        f"local elev = World_GetElevation({{latitude={lat}, longitude={lon}}}) "
        f"if elev <= 0 then "
        f"ScenEdit_SetUnit({{"
        f"unitname='{unit_name}', "
        f"side='{side}', "
        f"course={{ {{latitude={lat}, longitude={lon}}} }} "
        f"}}) "
        f"return {{status='new waypoint set'}} "
        f"end "
        f"end "
        f"return {{status='waypoint already active'}} "
        f"end)()"
    )

def GetUnitCoords(unit_name: str, side: str) -> str:
    return (
        f"(function() "
        f"local u = ScenEdit_GetUnit({{unitname='{unit_name}', side='{side}'}}); "
        f"if u then "
        f"return {{ latitude = u.latitude, longitude = u.longitude }} "
        f"else "
        f"return {{ error = 'unit not found' }} "
        f"end "
        f"end)()"
    )

def GetAllyUnitCoords(unit_name: str, side: str) -> str:
    return (
        f"(function() "
        f"local result = {{}} "
        f"local a = VP_GetSide({{side = '{side}'}}) "
        f"for i, u in pairs(a.units) do "
        f"if u.name ~= '{unit_name}' then "
        f"local full = ScenEdit_GetUnit({{guid = u.guid}}) "
        f"if full then "
        f"table.insert(result, {{name = u.name, latitude = full.latitude, longitude = full.longitude}}) "
        f"end "
        f"end "
        f"end "
        f"return result "
        f"end)()"
    )

def GetContactCoords(side: str) -> str:
    return (
        f"(function() "
        f"local contacts = ScenEdit_GetContacts('{side}') "
        f"local msg = '' "
        f"if contacts then "
        f"for _, contact in pairs(contacts) do "
        f"msg = msg .. 'Checking contact: ' .. tostring(contact.name) .. "
        f"' | Latitude: ' .. tostring(contact.latitude) .. "
        f"' | Longitude: ' .. tostring(contact.longitude) .. '\' "
        f"end "
        f"return {{result=msg}} "
        f"end "
        f"return {{result='No contacts coords found'}} "
        f"end)()"
    )

def GetUnitCoordsByGuid(guid: str) -> str:
    return (
        f"(function() "
        f"local u = ScenEdit_GetUnit({{guid='{guid}'}}) "
        f"if u then "
        f"return {{latitude=u.latitude, longitude=u.longitude}} "
        f"end "
        f"return {{error='unit not found'}} "
        f"end)()"
    )


# Actions --------------------------------------------


def EnableShipRadar(unit_name:str, state: str) -> str:
    return (
        f"(function() "
        f"ScenEdit_SetEMCON("
        f"'Unit',"
        f"'{unit_name}', "
        f"'Radar={state}') "
        f"end)()"
    )

def GetContact(side: str):
    return (
        f"(function() "
        f"local contacts = ScenEdit_GetContacts('{side}') "
        f"local msg = '' "
        f"if contacts then "
        f"for _, contact in pairs(contacts) do "
        f"msg = msg .. 'Checking contact: ' .. tostring(contact.name) .. "
        f"' | Posture: ' .. tostring(contact.posture) .. " # 0 = Neutral (N), 1 = Friendly (F), 2 = Unfriendly (U), 3 = Hostile (H), 4 = Unknown (X)
        f"' | GUID: ' .. tostring(contact.actualunitid) .. '\\n' "
        f"end "
        f"return {{result=msg}} "
        f"end "
        f"return {{result='No contacts found'}} "
        f"end)()"
    )

def AttackEnemyContact(unit_name: str, side: str, target_guid: str = "") -> str: 
    return (
        f"(function() "
        f"local contacts = ScenEdit_GetContacts('{side}') "
        f"if contacts == nil then "
        f"return {{status='no contacts'}} "
        f"end "
        f"for _, contact in pairs(contacts) do "
        f"if '{target_guid}' == '' or contact.actualunitid == '{target_guid}' then "
        f"ScenEdit_AttackContact('{unit_name}', contact.guid, {{mode=0}}) "
        f"return {{status='attacking', target=tostring(contact.name)}} "
        f"end "
        f"end "
        f"return {{status='no contact found'}} "
        f"end)()"
    )


 #  Scenario ----------------------------


def GetRemainingTime() -> str:
    return (
        f"(function() "
        f"local s = VP_GetScenario() "
        f"local remaining = (tonumber(s.StartTimeNum) + tonumber(s.DurationNum)) - tonumber(s.CurrentTimeNum) "
        f"local h = math.floor(remaining / 3600) "
        f"local m = math.floor((remaining % 3600) / 60) "
        f"local sec = math.floor(remaining % 60) "
        f"return {{remaining=remaining, formatted=string.format('%02d:%02d:%02d', h, m, sec)}} "
        f"end)()"
    )

def GetScenarioStatus():
    return (
        f"(function() "
        f"local status = VP_GetScenario() "
        f"return {{status.Status, status.GameStatus}} "
        f"end)()"
    )

def RestartScenario(xml: str) -> str:
    import re
    
    Xml = xml.strip().replace('\n', ' ').replace('\r', ' ')
    return (
        f"(function() "
        f"local a = ScenEdit_ImportScenarioFromXML({{XML = [=[{Xml}]=]}}) "
        f"return {{ok=tostring(ok)}} "
        f"end)()"
    )

def Score(side: str) -> str:
    return (
        f"(function() "
        f"local score = ScenEdit_GetScore('{side}') "
        f"return {{score=score}} "
        f"end)()"
    )

def SetScore(side: str, score: int) -> str:
    return (
        f"(function() "
        f"ScenEdit_SetScore('{side}', {score}) "
        f"end)()"
    )


# Mission --------------------------------


def GetMissionType(side: str) -> str:
    return (
        f"(function() "
        f"local a = ScenEdit_GetMissions('{side}') "
        f"return {{a[1].type}} "
        f"end)()"
    )

def GetTarget(side: str) -> str:
    return (
        f"(function()"
        f"local a = ScenEdit_GetMissions('{side}')"
        f"return {{a[1].targetlist}}"
        f"end)()"
    )


# State ----------------------------------


def ChangeTimeCompression(number: int) -> str:
    return (
        f"(function() "
        f"VP_SetTimeCompression({number}) " # 0 = 1 sec, 1 = 2 sec, 2 = 5 sec, 3 = 15 sec, 4 = 30 sec, 5 = 60 min 
        f"end)()"
    )

def GetDamage(unit_name: str) -> str:
    return (
        f"(function() "
        f"local u = ScenEdit_GetUnit({{unitname = '{unit_name}'}}) "
        f"return {{'Damage Precent: ' .. u.damage.dp_percent, 'Fires: ' .. u.damage.fires}}"
        f"end)()"
    )

def SetSpeed(unit_name: str, throttle: int) -> str:
    return ( 
        f"(function()"
        f"ScenEdit_SetUnit({{unitname='{unit_name}',manualthrottle={throttle}}})" # 0 = FullStop, 1 = Creep, 2 = Cruise, 3 = Full, 4 = Flank, 5  = None 
    )

def GetAllyAlive(unit_name: str, side: str) -> str:
    return (
        f"(function() "
        f"local result = {{}} "
        f"local a = VP_GetSide({{side = '{side}'}}) "
        f"for _, u in pairs(a.units) do "
        f"if u.name ~= '{unit_name}' then "
        f"result[#result + 1] = {{name = u.name, guid = u.guid, alive= true}} "
        f"end "
        f"end "
        f"if #result == 0 then "
        f"return {{alive= false}} "
        f"end "
        f"return result "
        f"end)()"
    )

def GetEnemyRange() -> str:
    return (
        f"(function() "
        f"u = ScenEdit_GetUnit({{side = 'Red', unitname = 'RED_SSM1'}}) "
        f"for i, sensors in pairs(u.sensors) do "
        f"return {{sensors.sensor_maxrange}} "
        f"end)()"
    )


# Other -----------------------------------


def GetControllableUnit(side: str) ->str:
    return (
        f"(function() "
        f"local a = VP_GetSide({{side = '{side}'}}) "
        f"if not a or not a.units then "
        f"return {{error='side or units not found'}} "
        f"end "
        f"local preferred_units = {{'F_801', 'CSGN_9_Long_Beach', 'CGN 9 Long Beach', 'F 802 De Zeven Provinciën'}} "
        f"for _, pname in ipairs(preferred_units) do "
        f"for _, unit in pairs(a.units) do "
        f"if unit.name == pname then "
        f"return {{unitname=unit.name}} "
        f"end "
        f"end "
        f"end "
        f"return {{status='no unit found'}} "
        f"end)()"
    )

