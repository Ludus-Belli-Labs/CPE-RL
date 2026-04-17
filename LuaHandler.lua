


-- Test commands --------------------------------

function GetTitle()
    return ScenEdit_GetScenarioTitle()
end

function GetSelectedUnits()
    return ScenEdit_SelectedUnits()
end

function StartSimulation()
    VP_RunSimulation()
end


-- Movement ----------------------------------------

function SetUnitWaypoint()
    local u = ScenEdit_GetUnit({unitname='{unit_name}', side='{side}'})
    if u and (#u.course == 0) then
        local elev = World_GetElevation({latitude={lat}, longitude={lon}})
        if elev <= 0 then
            ScenEdit_SetUnit({
                unitname='{unit_name}',
                side='{side}',
                course={ {latitude={lat}, longitude={lon}} }
            })
            return {status='new waypoint set'}
        end
    end
    return {status='waypoint already active'}
end

function GetUnitCoords()
    local u = ScenEdit_GetUnit({unitname='{unit_name}', side='{side}'})
    if u then
        return { latitude = u.latitude, longitude = u.longitude }
    else
        return { error = 'unit not found' }
    end
end

function GetAllyUnitCoords()
    local result = {}
    local a = VP_GetSide({side = '{side}'})
    for i, u in pairs(a.units) do
        if u.name ~= '{unit_name}' then
            local full = ScenEdit_GetUnit({guid = u.guid})
            if full then
                table.insert(result, {name = u.name, latitude = full.latitude, longitude = full.longitude})
            end
        end
    end
    return result
end

function GetContactCoords()
    local contacts = ScenEdit_GetContacts('{side}')
    local msg = ''
    if contacts then
        for _, contact in pairs(contacts) do
            msg = msg .. 'Checking contact: ' .. tostring(contact.name) ..
                ' | Latitude: ' .. tostring(contact.latitude) ..
                ' | Longitude: ' .. tostring(contact.longitude) .. '\n'
        end
        return {result=msg}
    end
    return {result='No contacts coords found'}
end


-- Actions --------------------------------------------

function EnableShipRadar()
    -- state: 'Active' or 'Passive'
    ScenEdit_SetEMCON('Unit', '{unit_name}', 'Radar={state}')
end

function GetContact()
    local contacts = ScenEdit_GetContacts('{side}')
    local msg = ''
    if contacts then
        for _, contact in pairs(contacts) do
            -- Posture: 0=Neutral(N), 1=Friendly(F), 2=Unfriendly(U), 3=Hostile(H), 4=Unknown(X)
            msg = msg .. 'Checking contact: ' .. tostring(contact.name) ..
                ' | Posture: ' .. tostring(contact.posture) ..
                ' | GUID: ' .. tostring(contact.actualunitid) .. '\n'
        end
        return {result=msg}
    end
    return {result='No contacts found'}
end

function AttackEnemyContact()
    local contacts = ScenEdit_GetContacts('{side}')
    if contacts == nil then
        return {status='no contacts'}
    end
    for _, contact in pairs(contacts) do
        if '{target_guid}' == '' or contact.actualunitid == '{target_guid}' then
            ScenEdit_AttackContact('{unit_name}', contact.guid, {mode=0})
            return {status='attacking', target=tostring(contact.name)}
        end
    end
    return {status='no contact found'}
end


-- Scenario ----------------------------

function GetRemainingTime()
    local s = VP_GetScenario()
    local remaining = (tonumber(s.StartTimeNum) + tonumber(s.DurationNum)) - tonumber(s.CurrentTimeNum)
    local h = math.floor(remaining / 3600)
    local m = math.floor((remaining % 3600) / 60)
    local sec = math.floor(remaining % 60)
    return {remaining=remaining, formatted=string.format('%02d:%02d:%02d', h, m, sec)}
end

function GetScenarioStatus()
    local status = VP_GetScenario()
    return {status.Status, status.GameStatus}
end

function RestartScenario()
    ScenEdit_ImportScenarioFromXML({XML = [[{xml}]]})
end

function Score()
    local score = ScenEdit_GetScore('{side}')
    return {score=score}
end


-- Mission --------------------------------

function GetMissionType()
    local a = ScenEdit_GetMissions('{side}')
    return {a[1].type}
end

function GetTarget()
    local a = ScenEdit_GetMissions('{side}')
    return {a[1].targetlist}
end


-- State ----------------------------------

function ChangeTimeCompression()
    -- number: 0=1s, 1=2s, 2=5s, 3=15s, 4=30s, 5=60min
    VP_SetTimeCompression({number})
end

function GetDamage()
    local u = ScenEdit_GetUnit({unitname = '{unit_name}'})
    return {'Damage Percent: ' .. u.damage.dp_percent, 'Fires: ' .. u.damage.fires}
end

function SetSpeed()
    -- throttle: 0=FullStop, 1=Creep, 2=Cruise, 3=Full, 4=Flank, 5=None
    ScenEdit_SetUnit({unitname='{unit_name}', throttle={throttle}})
end

function GetAllyAlive()
    local result = {}
    local a = VP_GetSide({side = '{side}'})
    for _, u in pairs(a.units) do
        if u.name ~= '{unit_name}' then
            result[#result + 1] = {name = u.name, guid = u.guid, alive = true}
        end
    end
    if #result == 0 then
        return {alive = false}
    end
    return result
end


-- Other -----------------------------------

function GetControllableUnit()
    local a = VP_GetSide({side = '{side}'})
    if not a or not a.units then
        return {error='side or units not found'}
    end
    local preferred_units = {'F_801', 'CSGN_9_Long_Beach'}
    for _, pname in ipairs(preferred_units) do
        for _, unit in pairs(a.units) do
            if unit.name == pname then
                return {unitname=unit.name}
            end
        end
    end
    return {status='no unit found'}
end