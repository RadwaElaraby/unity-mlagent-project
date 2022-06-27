using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Area : MonoBehaviour
{
    List<Item> items = new List<Item>();
    List<Target> targets = new List<Target>();

    public List<RLAgent> rlAgents = new List<RLAgent>();

    public GameObject itemPrefab;
    public GameObject targetPrefab;

    public int noTargets = 1;
    public int noItemsPerTarget = 2;
    [HideInInspector]
    public int noTasks;
    [HideInInspector]
    public int initialNoTargets;
    [HideInInspector]
    public int initialNoItemsPerTarget;

    Floor floor;
    [HideInInspector]
    public BoxCollider floorCollider;

    public float minDistanceFromTarget;
    public float maxDistanceFromTarget;


    //[HideInInspector]
    //public SimpleMultiAgentGroup MultiAgentGroup;

    private void Start()
    {
        floor = GetComponentInChildren<Floor>();
        floorCollider = floor.GetComponent<BoxCollider>();

        foreach (var a in GetComponentsInChildren<RLAgent>())
        {
            rlAgents.Add(a);
        }

        initialNoTargets = noTargets;
        initialNoItemsPerTarget = noItemsPerTarget;
        noTasks = initialNoTargets * initialNoItemsPerTarget;

        ResetScene();
    }

    private void FixedUpdate()
    {
        if (noTasks == 0)
        {
            foreach (var a in rlAgents)
            {
                a.EndEpisode();
            }

            ResetScene();
        }

        if (rlAgents[0].StepCount >= rlAgents[0].MaxStep-1)
        {
            ResetScene();
        }
    }

    public void ResetScene()
    {
        RemoveExistingTargetsAndItems();

        noTargets = initialNoTargets;
        noItemsPerTarget = initialNoItemsPerTarget;
        noTasks = noItemsPerTarget * noTargets;

        for (int i = 0; i < noTargets; i++)
        {
            var targetPos = GetTargetPosition();
            var targetObj = Instantiate(targetPrefab, targetPos + transform.position, Quaternion.identity, transform);

            var material = getMaterial(i);
            targetObj.GetComponent<Renderer>().material = material;

            var target = targetObj.GetComponent<Target>();
            targets.Add(target);
            var targetCollider = target.GetComponent<BoxCollider>();

            for (int j = 0; j < noItemsPerTarget; j++)
            {
                var potentialPos = GetItemPosition(targetCollider);
                var itemObj = Instantiate(itemPrefab, potentialPos, Quaternion.identity, transform);
                itemObj.GetComponent<Renderer>().material = material;
                var item = itemObj.GetComponent<Item>();
                item.target = target;
                items.Add(item);
            }
        }
    }

    private void RemoveExistingTargetsAndItems()
    {
        foreach (var item in items)
        {
            Destroy(item.gameObject);
        }
        items.Clear();

        foreach (var target in targets)
        {
            Destroy(target.gameObject);
        }
        targets.Clear();
    }

    private Vector3 GetTargetPosition()
    {
        float targetScaleX = 0.1f * 10;
        float targetScaleZ = 0.1f * 10;
        float targetMaxPosX = floorCollider.bounds.extents.x / 2 - targetScaleX / 2;
        float targetMaxPosZ = floorCollider.bounds.extents.z / 2 - targetScaleZ / 2;

        var targetPosX = Random.Range(-targetMaxPosX, targetMaxPosX);
        var targetPosZ = Random.Range(-targetMaxPosZ, targetMaxPosZ);
        var targetPos = new Vector3(targetPosX, 0, targetPosZ);

        targetPos.y = 0.1f;
        return targetPos;
        //return new Vector3(0, 0.1f, -10.6f);
    }

    private Vector3 GetItemPosition(BoxCollider targetCollider)
    {
        Vector3 potentialPos;

        //float itemScaleX = 0.25f;
        //float itemtScaleZ = 0.25f;
        //float itemMaxPosX = FloorCollider.bounds.extents.x - itemScaleX / 2;
        //float itemMaxPosZ = FloorCollider.bounds.extents.z - itemtScaleZ / 2;

        float itemMinPosX;
        float itemMaxPosX;
        // south
        if (Random.Range(0f, 1f) > 0.5f)
        {
            itemMinPosX = Mathf.Max(targetCollider.bounds.min.x - maxDistanceFromTarget, floorCollider.bounds.min.x);
            itemMaxPosX = targetCollider.bounds.min.x - minDistanceFromTarget;
        }
        // north
        else
        {
            itemMinPosX = targetCollider.bounds.max.x + minDistanceFromTarget;
            itemMaxPosX = Mathf.Min(targetCollider.bounds.max.x + maxDistanceFromTarget, floorCollider.bounds.max.x);
        }

        float itemMinPosZ;
        float itemMaxPosZ;
        // west
        if (Random.Range(0f, 1f) > 0.5f)
        {
            itemMinPosZ = Mathf.Max(targetCollider.bounds.min.z - maxDistanceFromTarget, floorCollider.bounds.min.z);
            itemMaxPosZ = targetCollider.bounds.min.z - minDistanceFromTarget;
        }
        // north
        else
        {
            itemMinPosZ = targetCollider.bounds.max.z + minDistanceFromTarget;
            itemMaxPosZ = Mathf.Min(targetCollider.bounds.max.z + maxDistanceFromTarget, floorCollider.bounds.max.z);
        }

        var itemPosX = Random.Range(itemMinPosX, itemMaxPosX);
        var itemPosZ = Random.Range(itemMinPosZ, itemMaxPosZ);
        potentialPos = new Vector3(itemPosX, 1f, itemPosZ);

        //potentialPos.y = 1f;
        return potentialPos;
    }

    private Material getMaterial(int i)
    {
        var material_name = "material_" + i;
        var material = Resources.Load(material_name, typeof(Material)) as Material;

        return material;
    }

    /*
    private Vector3 GetItemPositionNew(BoxCollider targetCollider)
    {
        Vector3 potentialPos;

        //float itemScaleX = 0.25f;
        //float itemtScaleZ = 0.25f;
        //float itemMaxPosX = FloorCollider.bounds.extents.x - itemScaleX / 2;
        //float itemMaxPosZ = FloorCollider.bounds.extents.z - itemtScaleZ / 2;

        var itemMinPosX = floorCollider.bounds.min.x;
        var itemMaxPosX = floorCollider.bounds.max.x;
        var itemMinPosZ = targetCollider.bounds.max.z;
        var itemMaxPosZ = floorCollider.bounds.max.z;

        var itemPosX = Random.Range(itemMinPosX, itemMaxPosX);
        var itemPosZ = Random.Range(itemMinPosZ, itemMaxPosZ);
        potentialPos = new Vector3(itemPosX, 1f, itemPosZ);

        //potentialPos.y = 1f;
        return potentialPos;
    }*/


    /// <summary>
    /// 
    /// </summary>
    /// <returns></returns>
    /// 
    /*
    public Item FindNearestItem()
    {
        Item nearestItem = null;
        var nearestItemDistance = Mathf.Infinity;

        for (var i = 0; i < items.Count; i++)
        {
            var currentItem = items[i];
            if (currentItem.gameObject.activeSelf)
            {
                var distance = Vector3.Distance(currentItem.transform.position, rlAgent.transform.position);
                if (distance < nearestItemDistance)
                {
                    nearestItemDistance = distance;
                    nearestItem = currentItem;
                }
            }
        }

        return nearestItem;
    }
    */
}
