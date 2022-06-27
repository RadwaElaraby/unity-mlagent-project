using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Target : MonoBehaviour
{
    private Area area;
    private Material targetMaterial;

    private void Start()
    {
        area = GetComponentInParent<Area>();
        targetMaterial = GetComponent<Renderer>().material;
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.tag == "item")
        {
            var Item = other.GetComponent<Item>();
            var ItemMaterial = Item.GetComponent<Renderer>().material;

            if (ItemMaterial != null && targetMaterial != null && ItemMaterial.name == targetMaterial.name)
            {
                //Area.MultiAgentGroup.AddGroupReward(10);
                area.noTasks -= 1;
                Item.gameObject.SetActive(false);

                foreach(var a in area.rlAgents)
                {
                    a.ScoredAGoal();
                }
            }
        }
    }
}
