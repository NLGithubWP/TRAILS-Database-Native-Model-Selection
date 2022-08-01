package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

type archInfo struct {
	ArchitectureId      string           `json:"architecture_id"`
	Scores              map[string]SInfo `json:"scores"`
	TestAccuracy        string           `json:"test_accuracy"`
	TrainAccuracy       string           `json:"train_accuracy"`
	TrainableParameters string           `json:"trainable_parameters"`
	TrainingTime        string           `json:"training_time"`
	ValidationAccuracy  string           `json:"validation_accuracy"`
}

type Score struct {
	GradNorm       SInfo `json:"grad_norm"`
	Fisher         SInfo `json:"fisher"`
	GradPlain      SInfo `json:"grad_plain"`
	Grasp          SInfo `json:"grasp"`
	JacobConv      SInfo `json:"jacob_conv"`
	NasWot         SInfo `json:"nas_wot"`
	NtkCondNum     SInfo `json:"ntk_cond_num"`
	NtkTrace       SInfo `json:"ntk_trace"`
	NtkTraceApprox SInfo `json:"ntk_trace_approx"`
	Snip           SInfo `json:"snip"`
	Synflow        SInfo `json:"synflow"`
	WeightNorm     SInfo `json:"weight_norm"`
}

type SInfo struct {
	Score     string `json:"score"`
	TimeUsage string `json:"time_usage"`
}

func readJson(file_res string) map[string]archInfo {
	fileContent, err := os.Open(file_res)
	if err != nil {
		log.Fatal(err)
	}
	defer fileContent.Close()
	byteResult, _ := ioutil.ReadAll(fileContent)
	var res map[string]archInfo
	err = json.Unmarshal(byteResult, &res)
	if err != nil {
		fmt.Println(err)
	}
	return res
}

func compare_score_201_bn(m_name string, s1 float64, s2 float64) bool {
	if m_name == "grad_norm" {
		return s1 > s2
	}
	if m_name == "grad_plain" {
		return s1 < s2
	}
	if m_name == "ntk_cond_num" {
		return s1 < s2
	}
	if m_name == "ntk_trace" {
		return s1 > s2
	}
	if m_name == "ntk_trace_approx" {
		return s1 > s2
	}
	if m_name == "fisher" {
		return s1 > s2
	}
	if m_name == "grasp" {
		return s1 > s2
	}
	if m_name == "snip" {
		return s1 > s2
	}
	if m_name == "synflow" {
		return s1 > s2
	}
	if m_name == "weight_norm" {
		return s1 > s2
	}
	if m_name == "nas_wot" {
		return s1 > s2
	}
	return true
}

func compare_score_201_NO_bn(m_name string, s1 float64, s2 float64) bool {
	if m_name == "grad_norm" {
		return s1 < s2
	}
	if m_name == "grad_plain" {
		return s1 < s2
	}
	if m_name == "ntk_cond_num" {
		return s1 < s2
	}
	if m_name == "ntk_trace" {
		return s1 < s2
	}
	if m_name == "ntk_trace_approx" {
		return s1 < s2
	}
	if m_name == "fisher" {
		return s1 < s2
	}
	if m_name == "grasp" {
		return s1 > s2
	}
	if m_name == "snip" {
		return s1 < s2
	}
	if m_name == "synflow" {
		return s1 > s2
	}
	if m_name == "weight_norm" {
		return s1 < s2
	}
	if m_name == "nas_wot" {
		return s1 > s2
	}
	return true
}

func compare_score_101_bn(m_name string, s1 float64, s2 float64) bool {
	if m_name == "grad_norm" {
		return s1 < s2
	}
	if m_name == "grad_plain" {
		return s1 < s2
	}
	if m_name == "ntk_cond_num" {
		return s1 < s2
	}
	if m_name == "ntk_trace" {
		return s1 < s2
	}
	if m_name == "ntk_trace_approx" {
		return s1 < s2
	}
	if m_name == "fisher" {
		return s1 < s2
	}
	if m_name == "grasp" {
		return s1 > s2
	}
	if m_name == "snip" {
		return s1 < s2
	}
	if m_name == "synflow" {
		return s1 > s2
	}
	if m_name == "weight_norm" {
		return s1 > s2
	}
	if m_name == "nas_wot" {
		return s1 > s2
	}
	return true
}

func compare_score_101_NO_bn(m_name string, s1 float64, s2 float64) bool {
	if m_name == "grad_norm" {
		return s1 < s2
	}
	if m_name == "grad_plain" {
		return s1 < s2
	}
	if m_name == "ntk_cond_num" {
		return s1 < s2
	}
	if m_name == "ntk_trace" {
		return s1 < s2
	}
	if m_name == "ntk_trace_approx" {
		return s1 < s2
	}
	if m_name == "fisher" {
		return s1 < s2
	}
	if m_name == "grasp" {
		return s1 > s2
	}
	if m_name == "snip" {
		return s1 < s2
	}
	if m_name == "synflow" {
		return s1 > s2
	}
	if m_name == "weight_norm" {
		return s1 > s2
	}
	if m_name == "nas_wot" {
		return s1 > s2
	}
	return true
}

func compare_score_201_union_best(m_name string, s1 float64, s2 float64) bool {
	if m_name == "grad_norm" {
		return s1 > s2
	}
	if m_name == "grad_plain" {
		return s1 < s2
	}
	if m_name == "ntk_cond_num" {
		return s1 < s2
	}
	if m_name == "ntk_trace" {
		return s1 < s2
	}
	if m_name == "ntk_trace_approx" {
		return s1 < s2
	}
	if m_name == "fisher" {
		return s1 < s2
	}
	if m_name == "grasp" {
		return s1 > s2
	}
	if m_name == "snip" {
		return s1 > s2
	}
	if m_name == "synflow" {
		return s1 > s2
	}
	if m_name == "nas_wot" {
		return s1 > s2
	}
	return true
}

func vote_between(value1 archInfo, value2 archInfo, metricsList []string, spaceName string) string {
	leftVote := 0
	rightVote := 0
	var funcUsed func(m_name string, s1 float64, s2 float64) bool
	if spaceName == "101bn" {
		funcUsed = compare_score_101_bn
	} else if spaceName == "101nobn" {
		funcUsed = compare_score_101_NO_bn
	} else if spaceName == "201bn" {
		funcUsed = compare_score_201_bn
	} else if spaceName == "201nobn" {
		funcUsed = compare_score_201_NO_bn
	} else if spaceName == "201union_best" {
		funcUsed = compare_score_201_union_best
	} else {
		fmt.Println("space cannot recognized")
	}

	for _, algName := range metricsList {
		var s1 float64
		var s2 float64
		s1, err1 := strconv.ParseFloat(value1.Scores[algName].Score, 64)
		s2, err2 := strconv.ParseFloat(value2.Scores[algName].Score, 64)
		if err1 != nil || err2 != nil {
			fmt.Println(err1)
			fmt.Println(err2)
			return err1.Error()
		}

		if funcUsed(algName, s1, s2) {
			leftVote += 1
		} else {
			rightVote += 1
		}
	}

	if leftVote > rightVote {
		return value1.ArchitectureId
	} else {
		return value2.ArchitectureId
	}
}

func checkVote(data map[string]archInfo, spaceName string, voteList [][]string, fileName string, archList []string) {
	all_ss := []string{
		"grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx",
		"fisher", "grasp", "snip", "synflow"}

	visited := make(map[string]struct{})
	total_pair := 0
	vote_right := make(map[int]int)
	num_archs := make(map[string]struct{})

	for index, _ := range voteList {
		vote_right[index] = 0
	}

	logIndex := 0
	for _, archID1 := range archList {
		value1 := data[archID1]
		for archID2, value2 := range data {
			value1.ArchitectureId = archID1
			value2.ArchitectureId = archID2

			if value1.TestAccuracy == value2.TestAccuracy {
				continue
			}

			var arch_vsted string
			if archID1 < archID2 {
				arch_vsted = archID1 + "_" + archID2
			} else {
				arch_vsted = archID2 + "_" + archID1
			}

			if _, ok := visited[arch_vsted]; ok {
				continue
			}

			isContinue := 1
			for _, algName := range all_ss {
				if _, ok := value1.Scores[algName]; !ok {
					isContinue = 0
					break
				}
				if _, ok := value2.Scores[algName]; !ok {
					isContinue = 0
					break
				}
			}
			if isContinue == 0 {
				continue
			}

			visited[arch_vsted] = struct{}{}

			//fmt.Println(unsafe.Sizeof(arch_vsted))
			//fmt.Println(unsafe.Sizeof(visited))

			for index, metricsList := range voteList {

				var v1_taccu float64
				var v2_taccu float64

				v1_taccu, err1 := strconv.ParseFloat(value1.TestAccuracy, 64)
				v2_taccu, err2 := strconv.ParseFloat(value2.TestAccuracy, 64)
				if err1 != nil || err2 != nil {
					fmt.Println(err1)
					fmt.Println(err2)
					return
				}

				if v1_taccu > v2_taccu && vote_between(value1, value2, metricsList, spaceName) == value1.ArchitectureId {
					vote_right[index] += 1
				}

				if v1_taccu < v2_taccu && vote_between(value1, value2, metricsList, spaceName) == value2.ArchitectureId {
					vote_right[index] += 1
				}
			}
			total_pair += 1
		}
		num_archs[archID1] = struct{}{}
		logIndex += 1
		if logIndex%10 == 0 {
			fmt.Printf("%s evaluate %d archs \n", fileName, len(num_archs))
		}
	}

	filePath := "./" + fileName
	file, err := os.OpenFile(filePath, os.O_WRONLY|os.O_CREATE, 0666)
	if err != nil {
		fmt.Println("文件打开失败", err)
	}
	defer file.Close()
	write := bufio.NewWriter(file)
	write.WriteString(toStr(num_archs))
	write.WriteString(fmt.Sprintf("%s total_pair = %d \n", spaceName, total_pair))
	for index, ele := range vote_right {
		write.WriteString(fmt.Sprintf("%s, [%s], %d / %d \n", spaceName, strings.Join(voteList[index], ","), ele, total_pair))
	}

	//Flush将缓存的文件真正写入到文件中
	write.Flush()
	write = nil
}

func checkAllVote(data map[string]archInfo, spaceName string, voteList [][]string, fileName string, archList []string) {
	total_pair := 0
	vote_right := make(map[int]int)

	for index, _ := range voteList {
		vote_right[index] = 0
	}

	for _, archIDCombines := range archList {

		arch_list := strings.Split(archIDCombines, "__")

		archID1 := arch_list[0]
		archID2 := arch_list[1]
		value1 := data[archID1]
		value2 := data[archID2]
		value1.ArchitectureId = archID1
		value2.ArchitectureId = archID2

		if value1.TestAccuracy == value2.TestAccuracy {
			continue
		}

		for index, metricsList := range voteList {
			var v1_taccu float64
			var v2_taccu float64
			v1_taccu, err1 := strconv.ParseFloat(value1.TestAccuracy, 64)
			v2_taccu, err2 := strconv.ParseFloat(value2.TestAccuracy, 64)
			if err1 != nil || err2 != nil {
				fmt.Println(err1)
				fmt.Println(err2)
				return
			}
			if v1_taccu > v2_taccu && vote_between(value1, value2, metricsList, spaceName) == value1.ArchitectureId {
				vote_right[index] += 1
			}
			if v1_taccu < v2_taccu && vote_between(value1, value2, metricsList, spaceName) == value2.ArchitectureId {
				vote_right[index] += 1
			}
		}
		total_pair += 1
	}
	filePath := "./" + fileName
	file, err := os.OpenFile(filePath, os.O_WRONLY|os.O_CREATE, 0666)
	if err != nil {
		fmt.Println("文件打开失败", err)
	}
	defer file.Close()
	write := bufio.NewWriter(file)
	write.WriteString(fmt.Sprintf("%s total_pair = %d \n", spaceName, total_pair))
	for index, ele := range vote_right {
		write.WriteString(fmt.Sprintf("%s, [%s], %d / %d \n", spaceName, strings.Join(voteList[index], ","), ele, total_pair))
	}
	write.Flush()
	write = nil
}

func toStr(m map[string]struct{}) string {
	var res string = "[]string{"
	for key, _ := range m {
		res += fmt.Sprintf("%s", key)
		res += ","
	}
	res += "}\n"
	return res
}

func getKeys(data map[string]archInfo) []string {
	var res []string
	for k, _ := range data {
		res = append(res, k)
	}
	return res

}

func getKeys2(data map[string]int) []string {
	var res []string
	for k, _ := range data {
		res = append(res, k)
	}
	return res
}

func measureOneDatasetSequence(data map[string]archInfo) {
	fmt.Println("Len of the bn key = " + fmt.Sprintf("%d", len(data)))

	grad_norm := []string{"grad_norm"}
	grad_plain := []string{"grad_plain"}
	ntk_cond_num := []string{"ntk_cond_num"}
	ntk_trace := []string{"ntk_trace"}
	ntk_trace_approx := []string{"ntk_trace_approx"}
	fisher := []string{"fisher"}
	grasp := []string{"grasp"}
	snip := []string{"snip"}
	synflow := []string{"synflow"}
	weight_norm := []string{"weight_norm"}
	nas_wot := []string{"nas_wot"}

	voteList := [][]string{
		grad_norm, grad_plain, ntk_cond_num, ntk_trace, ntk_trace_approx, grasp, fisher,
		snip, synflow, weight_norm, nas_wot}

	fmt.Println("Evaluate 201bn")
	start0 := time.Now()
	checkVote(data, "201bn", voteList, "201bnfile", getKeys(data))
	fmt.Printf("201bn main, execution time %s\n", time.Since(start0))
}

func measureAllVoteMetrics(data201UnionBest map[string]archInfo) {
	all_vote_metrics := [][]string{{"grad_norm", "grad_plain", "nas_wot"}, {"grad_norm", "grad_plain", "ntk_cond_num"}, {"grad_norm", "nas_wot", "ntk_cond_num"}, {"grad_plain", "nas_wot", "ntk_cond_num"}, {"grad_norm", "grad_plain", "ntk_trace"}, {"grad_norm", "nas_wot", "ntk_trace"}, {"grad_plain", "nas_wot", "ntk_trace"}, {"grad_norm", "ntk_cond_num", "ntk_trace"}, {"grad_plain", "ntk_cond_num", "ntk_trace"}, {"nas_wot", "ntk_cond_num", "ntk_trace"}, {"grad_norm", "grad_plain", "ntk_trace_approx"}, {"grad_norm", "nas_wot", "ntk_trace_approx"}, {"grad_plain", "nas_wot", "ntk_trace_approx"}, {"grad_norm", "ntk_cond_num", "ntk_trace_approx"}, {"grad_plain", "ntk_cond_num", "ntk_trace_approx"}, {"nas_wot", "ntk_cond_num", "ntk_trace_approx"}, {"grad_norm", "ntk_trace", "ntk_trace_approx"}, {"grad_plain", "ntk_trace", "ntk_trace_approx"}, {"nas_wot", "ntk_trace", "ntk_trace_approx"}, {"ntk_cond_num", "ntk_trace", "ntk_trace_approx"}, {"grad_norm", "grad_plain", "fisher"}, {"grad_norm", "nas_wot", "fisher"}, {"grad_plain", "nas_wot", "fisher"}, {"grad_norm", "ntk_cond_num", "fisher"}, {"grad_plain", "ntk_cond_num", "fisher"}, {"nas_wot", "ntk_cond_num", "fisher"}, {"grad_norm", "ntk_trace", "fisher"}, {"grad_plain", "ntk_trace", "fisher"}, {"nas_wot", "ntk_trace", "fisher"}, {"ntk_cond_num", "ntk_trace", "fisher"}, {"grad_norm", "ntk_trace_approx", "fisher"}, {"grad_plain", "ntk_trace_approx", "fisher"}, {"nas_wot", "ntk_trace_approx", "fisher"}, {"ntk_cond_num", "ntk_trace_approx", "fisher"}, {"ntk_trace", "ntk_trace_approx", "fisher"}, {"grad_norm", "grad_plain", "grasp"}, {"grad_norm", "nas_wot", "grasp"}, {"grad_plain", "nas_wot", "grasp"}, {"grad_norm", "ntk_cond_num", "grasp"}, {"grad_plain", "ntk_cond_num", "grasp"}, {"nas_wot", "ntk_cond_num", "grasp"}, {"grad_norm", "ntk_trace", "grasp"}, {"grad_plain", "ntk_trace", "grasp"}, {"nas_wot", "ntk_trace", "grasp"}, {"ntk_cond_num", "ntk_trace", "grasp"}, {"grad_norm", "ntk_trace_approx", "grasp"}, {"grad_plain", "ntk_trace_approx", "grasp"}, {"nas_wot", "ntk_trace_approx", "grasp"}, {"ntk_cond_num", "ntk_trace_approx", "grasp"}, {"ntk_trace", "ntk_trace_approx", "grasp"}, {"grad_norm", "fisher", "grasp"}, {"grad_plain", "fisher", "grasp"}, {"nas_wot", "fisher", "grasp"}, {"ntk_cond_num", "fisher", "grasp"}, {"ntk_trace", "fisher", "grasp"}, {"ntk_trace_approx", "fisher", "grasp"}, {"grad_norm", "grad_plain", "snip"}, {"grad_norm", "nas_wot", "snip"}, {"grad_plain", "nas_wot", "snip"}, {"grad_norm", "ntk_cond_num", "snip"}, {"grad_plain", "ntk_cond_num", "snip"}, {"nas_wot", "ntk_cond_num", "snip"}, {"grad_norm", "ntk_trace", "snip"}, {"grad_plain", "ntk_trace", "snip"}, {"nas_wot", "ntk_trace", "snip"}, {"ntk_cond_num", "ntk_trace", "snip"}, {"grad_norm", "ntk_trace_approx", "snip"}, {"grad_plain", "ntk_trace_approx", "snip"}, {"nas_wot", "ntk_trace_approx", "snip"}, {"ntk_cond_num", "ntk_trace_approx", "snip"}, {"ntk_trace", "ntk_trace_approx", "snip"}, {"grad_norm", "fisher", "snip"}, {"grad_plain", "fisher", "snip"}, {"nas_wot", "fisher", "snip"}, {"ntk_cond_num", "fisher", "snip"}, {"ntk_trace", "fisher", "snip"}, {"ntk_trace_approx", "fisher", "snip"}, {"grad_norm", "grasp", "snip"}, {"grad_plain", "grasp", "snip"}, {"nas_wot", "grasp", "snip"}, {"ntk_cond_num", "grasp", "snip"}, {"ntk_trace", "grasp", "snip"}, {"ntk_trace_approx", "grasp", "snip"}, {"fisher", "grasp", "snip"}, {"grad_norm", "grad_plain", "synflow"}, {"grad_norm", "nas_wot", "synflow"}, {"grad_plain", "nas_wot", "synflow"}, {"grad_norm", "ntk_cond_num", "synflow"}, {"grad_plain", "ntk_cond_num", "synflow"}, {"nas_wot", "ntk_cond_num", "synflow"}, {"grad_norm", "ntk_trace", "synflow"}, {"grad_plain", "ntk_trace", "synflow"}, {"nas_wot", "ntk_trace", "synflow"}, {"ntk_cond_num", "ntk_trace", "synflow"}, {"grad_norm", "ntk_trace_approx", "synflow"}, {"grad_plain", "ntk_trace_approx", "synflow"}, {"nas_wot", "ntk_trace_approx", "synflow"}, {"ntk_cond_num", "ntk_trace_approx", "synflow"}, {"ntk_trace", "ntk_trace_approx", "synflow"}, {"grad_norm", "fisher", "synflow"}, {"grad_plain", "fisher", "synflow"}, {"nas_wot", "fisher", "synflow"}, {"ntk_cond_num", "fisher", "synflow"}, {"ntk_trace", "fisher", "synflow"}, {"ntk_trace_approx", "fisher", "synflow"}, {"grad_norm", "grasp", "synflow"}, {"grad_plain", "grasp", "synflow"}, {"nas_wot", "grasp", "synflow"}, {"ntk_cond_num", "grasp", "synflow"}, {"ntk_trace", "grasp", "synflow"}, {"ntk_trace_approx", "grasp", "synflow"}, {"fisher", "grasp", "synflow"}, {"grad_norm", "snip", "synflow"}, {"grad_plain", "snip", "synflow"}, {"nas_wot", "snip", "synflow"}, {"ntk_cond_num", "snip", "synflow"}, {"ntk_trace", "snip", "synflow"}, {"ntk_trace_approx", "snip", "synflow"}, {"fisher", "snip", "synflow"}, {"grasp", "snip", "synflow"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace_approx"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_trace", "ntk_trace_approx"}, {"grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace", "ntk_trace_approx"}, {"grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx"}, {"grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "fisher"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_trace", "fisher"}, {"grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace", "fisher"}, {"grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace", "fisher"}, {"grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "fisher"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_trace_approx", "fisher"}, {"grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace_approx", "fisher"}, {"grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "fisher"}, {"grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "fisher"}, {"grad_norm", "grad_plain", "ntk_trace", "ntk_trace_approx", "fisher"}, {"grad_norm", "nas_wot", "ntk_trace", "ntk_trace_approx", "fisher"}, {"grad_plain", "nas_wot", "ntk_trace", "ntk_trace_approx", "fisher"}, {"grad_norm", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher"}, {"grad_plain", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher"}, {"nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "grasp"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_trace", "grasp"}, {"grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace", "grasp"}, {"grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace", "grasp"}, {"grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "grasp"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_trace_approx", "grasp"}, {"grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace_approx", "grasp"}, {"grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "grasp"}, {"grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "grasp"}, {"grad_norm", "grad_plain", "ntk_trace", "ntk_trace_approx", "grasp"}, {"grad_norm", "nas_wot", "ntk_trace", "ntk_trace_approx", "grasp"}, {"grad_plain", "nas_wot", "ntk_trace", "ntk_trace_approx", "grasp"}, {"grad_norm", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "grasp"}, {"grad_plain", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "grasp"}, {"nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "grasp"}, {"grad_norm", "grad_plain", "nas_wot", "fisher", "grasp"}, {"grad_norm", "grad_plain", "ntk_cond_num", "fisher", "grasp"}, {"grad_norm", "nas_wot", "ntk_cond_num", "fisher", "grasp"}, {"grad_plain", "nas_wot", "ntk_cond_num", "fisher", "grasp"}, {"grad_norm", "grad_plain", "ntk_trace", "fisher", "grasp"}, {"grad_norm", "nas_wot", "ntk_trace", "fisher", "grasp"}, {"grad_plain", "nas_wot", "ntk_trace", "fisher", "grasp"}, {"grad_norm", "ntk_cond_num", "ntk_trace", "fisher", "grasp"}, {"grad_plain", "ntk_cond_num", "ntk_trace", "fisher", "grasp"}, {"nas_wot", "ntk_cond_num", "ntk_trace", "fisher", "grasp"}, {"grad_norm", "grad_plain", "ntk_trace_approx", "fisher", "grasp"}, {"grad_norm", "nas_wot", "ntk_trace_approx", "fisher", "grasp"}, {"grad_plain", "nas_wot", "ntk_trace_approx", "fisher", "grasp"}, {"grad_norm", "ntk_cond_num", "ntk_trace_approx", "fisher", "grasp"}, {"grad_plain", "ntk_cond_num", "ntk_trace_approx", "fisher", "grasp"}, {"nas_wot", "ntk_cond_num", "ntk_trace_approx", "fisher", "grasp"}, {"grad_norm", "ntk_trace", "ntk_trace_approx", "fisher", "grasp"}, {"grad_plain", "ntk_trace", "ntk_trace_approx", "fisher", "grasp"}, {"nas_wot", "ntk_trace", "ntk_trace_approx", "fisher", "grasp"}, {"ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "grasp"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "snip"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_trace", "snip"}, {"grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace", "snip"}, {"grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace", "snip"}, {"grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "snip"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_trace_approx", "snip"}, {"grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace_approx", "snip"}, {"grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "snip"}, {"grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "snip"}, {"grad_norm", "grad_plain", "ntk_trace", "ntk_trace_approx", "snip"}, {"grad_norm", "nas_wot", "ntk_trace", "ntk_trace_approx", "snip"}, {"grad_plain", "nas_wot", "ntk_trace", "ntk_trace_approx", "snip"}, {"grad_norm", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "snip"}, {"grad_plain", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "snip"}, {"nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "snip"}, {"grad_norm", "grad_plain", "nas_wot", "fisher", "snip"}, {"grad_norm", "grad_plain", "ntk_cond_num", "fisher", "snip"}, {"grad_norm", "nas_wot", "ntk_cond_num", "fisher", "snip"}, {"grad_plain", "nas_wot", "ntk_cond_num", "fisher", "snip"}, {"grad_norm", "grad_plain", "ntk_trace", "fisher", "snip"}, {"grad_norm", "nas_wot", "ntk_trace", "fisher", "snip"}, {"grad_plain", "nas_wot", "ntk_trace", "fisher", "snip"}, {"grad_norm", "ntk_cond_num", "ntk_trace", "fisher", "snip"}, {"grad_plain", "ntk_cond_num", "ntk_trace", "fisher", "snip"}, {"nas_wot", "ntk_cond_num", "ntk_trace", "fisher", "snip"}, {"grad_norm", "grad_plain", "ntk_trace_approx", "fisher", "snip"}, {"grad_norm", "nas_wot", "ntk_trace_approx", "fisher", "snip"}, {"grad_plain", "nas_wot", "ntk_trace_approx", "fisher", "snip"}, {"grad_norm", "ntk_cond_num", "ntk_trace_approx", "fisher", "snip"}, {"grad_plain", "ntk_cond_num", "ntk_trace_approx", "fisher", "snip"}, {"nas_wot", "ntk_cond_num", "ntk_trace_approx", "fisher", "snip"}, {"grad_norm", "ntk_trace", "ntk_trace_approx", "fisher", "snip"}, {"grad_plain", "ntk_trace", "ntk_trace_approx", "fisher", "snip"}, {"nas_wot", "ntk_trace", "ntk_trace_approx", "fisher", "snip"}, {"ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "snip"}, {"grad_norm", "grad_plain", "nas_wot", "grasp", "snip"}, {"grad_norm", "grad_plain", "ntk_cond_num", "grasp", "snip"}, {"grad_norm", "nas_wot", "ntk_cond_num", "grasp", "snip"}, {"grad_plain", "nas_wot", "ntk_cond_num", "grasp", "snip"}, {"grad_norm", "grad_plain", "ntk_trace", "grasp", "snip"}, {"grad_norm", "nas_wot", "ntk_trace", "grasp", "snip"}, {"grad_plain", "nas_wot", "ntk_trace", "grasp", "snip"}, {"grad_norm", "ntk_cond_num", "ntk_trace", "grasp", "snip"}, {"grad_plain", "ntk_cond_num", "ntk_trace", "grasp", "snip"}, {"nas_wot", "ntk_cond_num", "ntk_trace", "grasp", "snip"}, {"grad_norm", "grad_plain", "ntk_trace_approx", "grasp", "snip"}, {"grad_norm", "nas_wot", "ntk_trace_approx", "grasp", "snip"}, {"grad_plain", "nas_wot", "ntk_trace_approx", "grasp", "snip"}, {"grad_norm", "ntk_cond_num", "ntk_trace_approx", "grasp", "snip"}, {"grad_plain", "ntk_cond_num", "ntk_trace_approx", "grasp", "snip"}, {"nas_wot", "ntk_cond_num", "ntk_trace_approx", "grasp", "snip"}, {"grad_norm", "ntk_trace", "ntk_trace_approx", "grasp", "snip"}, {"grad_plain", "ntk_trace", "ntk_trace_approx", "grasp", "snip"}, {"nas_wot", "ntk_trace", "ntk_trace_approx", "grasp", "snip"}, {"ntk_cond_num", "ntk_trace", "ntk_trace_approx", "grasp", "snip"}, {"grad_norm", "grad_plain", "fisher", "grasp", "snip"}, {"grad_norm", "nas_wot", "fisher", "grasp", "snip"}, {"grad_plain", "nas_wot", "fisher", "grasp", "snip"}, {"grad_norm", "ntk_cond_num", "fisher", "grasp", "snip"}, {"grad_plain", "ntk_cond_num", "fisher", "grasp", "snip"}, {"nas_wot", "ntk_cond_num", "fisher", "grasp", "snip"}, {"grad_norm", "ntk_trace", "fisher", "grasp", "snip"}, {"grad_plain", "ntk_trace", "fisher", "grasp", "snip"}, {"nas_wot", "ntk_trace", "fisher", "grasp", "snip"}, {"ntk_cond_num", "ntk_trace", "fisher", "grasp", "snip"}, {"grad_norm", "ntk_trace_approx", "fisher", "grasp", "snip"}, {"grad_plain", "ntk_trace_approx", "fisher", "grasp", "snip"}, {"nas_wot", "ntk_trace_approx", "fisher", "grasp", "snip"}, {"ntk_cond_num", "ntk_trace_approx", "fisher", "grasp", "snip"}, {"ntk_trace", "ntk_trace_approx", "fisher", "grasp", "snip"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "synflow"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_trace", "synflow"}, {"grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace", "synflow"}, {"grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace", "synflow"}, {"grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "synflow"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_trace_approx", "synflow"}, {"grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace_approx", "synflow"}, {"grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "synflow"}, {"grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "synflow"}, {"grad_norm", "grad_plain", "ntk_trace", "ntk_trace_approx", "synflow"}, {"grad_norm", "nas_wot", "ntk_trace", "ntk_trace_approx", "synflow"}, {"grad_plain", "nas_wot", "ntk_trace", "ntk_trace_approx", "synflow"}, {"grad_norm", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "synflow"}, {"grad_plain", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "synflow"}, {"nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "synflow"}, {"grad_norm", "grad_plain", "nas_wot", "fisher", "synflow"}, {"grad_norm", "grad_plain", "ntk_cond_num", "fisher", "synflow"}, {"grad_norm", "nas_wot", "ntk_cond_num", "fisher", "synflow"}, {"grad_plain", "nas_wot", "ntk_cond_num", "fisher", "synflow"}, {"grad_norm", "grad_plain", "ntk_trace", "fisher", "synflow"}, {"grad_norm", "nas_wot", "ntk_trace", "fisher", "synflow"}, {"grad_plain", "nas_wot", "ntk_trace", "fisher", "synflow"}, {"grad_norm", "ntk_cond_num", "ntk_trace", "fisher", "synflow"}, {"grad_plain", "ntk_cond_num", "ntk_trace", "fisher", "synflow"}, {"nas_wot", "ntk_cond_num", "ntk_trace", "fisher", "synflow"}, {"grad_norm", "grad_plain", "ntk_trace_approx", "fisher", "synflow"}, {"grad_norm", "nas_wot", "ntk_trace_approx", "fisher", "synflow"}, {"grad_plain", "nas_wot", "ntk_trace_approx", "fisher", "synflow"}, {"grad_norm", "ntk_cond_num", "ntk_trace_approx", "fisher", "synflow"}, {"grad_plain", "ntk_cond_num", "ntk_trace_approx", "fisher", "synflow"}, {"nas_wot", "ntk_cond_num", "ntk_trace_approx", "fisher", "synflow"}, {"grad_norm", "ntk_trace", "ntk_trace_approx", "fisher", "synflow"}, {"grad_plain", "ntk_trace", "ntk_trace_approx", "fisher", "synflow"}, {"nas_wot", "ntk_trace", "ntk_trace_approx", "fisher", "synflow"}, {"ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "synflow"}, {"grad_norm", "grad_plain", "nas_wot", "grasp", "synflow"}, {"grad_norm", "grad_plain", "ntk_cond_num", "grasp", "synflow"}, {"grad_norm", "nas_wot", "ntk_cond_num", "grasp", "synflow"}, {"grad_plain", "nas_wot", "ntk_cond_num", "grasp", "synflow"}, {"grad_norm", "grad_plain", "ntk_trace", "grasp", "synflow"}, {"grad_norm", "nas_wot", "ntk_trace", "grasp", "synflow"}, {"grad_plain", "nas_wot", "ntk_trace", "grasp", "synflow"}, {"grad_norm", "ntk_cond_num", "ntk_trace", "grasp", "synflow"}, {"grad_plain", "ntk_cond_num", "ntk_trace", "grasp", "synflow"}, {"nas_wot", "ntk_cond_num", "ntk_trace", "grasp", "synflow"}, {"grad_norm", "grad_plain", "ntk_trace_approx", "grasp", "synflow"}, {"grad_norm", "nas_wot", "ntk_trace_approx", "grasp", "synflow"}, {"grad_plain", "nas_wot", "ntk_trace_approx", "grasp", "synflow"}, {"grad_norm", "ntk_cond_num", "ntk_trace_approx", "grasp", "synflow"}, {"grad_plain", "ntk_cond_num", "ntk_trace_approx", "grasp", "synflow"}, {"nas_wot", "ntk_cond_num", "ntk_trace_approx", "grasp", "synflow"}, {"grad_norm", "ntk_trace", "ntk_trace_approx", "grasp", "synflow"}, {"grad_plain", "ntk_trace", "ntk_trace_approx", "grasp", "synflow"}, {"nas_wot", "ntk_trace", "ntk_trace_approx", "grasp", "synflow"}, {"ntk_cond_num", "ntk_trace", "ntk_trace_approx", "grasp", "synflow"}, {"grad_norm", "grad_plain", "fisher", "grasp", "synflow"}, {"grad_norm", "nas_wot", "fisher", "grasp", "synflow"}, {"grad_plain", "nas_wot", "fisher", "grasp", "synflow"}, {"grad_norm", "ntk_cond_num", "fisher", "grasp", "synflow"}, {"grad_plain", "ntk_cond_num", "fisher", "grasp", "synflow"}, {"nas_wot", "ntk_cond_num", "fisher", "grasp", "synflow"}, {"grad_norm", "ntk_trace", "fisher", "grasp", "synflow"}, {"grad_plain", "ntk_trace", "fisher", "grasp", "synflow"}, {"nas_wot", "ntk_trace", "fisher", "grasp", "synflow"}, {"ntk_cond_num", "ntk_trace", "fisher", "grasp", "synflow"}, {"grad_norm", "ntk_trace_approx", "fisher", "grasp", "synflow"}, {"grad_plain", "ntk_trace_approx", "fisher", "grasp", "synflow"}, {"nas_wot", "ntk_trace_approx", "fisher", "grasp", "synflow"}, {"ntk_cond_num", "ntk_trace_approx", "fisher", "grasp", "synflow"}, {"ntk_trace", "ntk_trace_approx", "fisher", "grasp", "synflow"}, {"grad_norm", "grad_plain", "nas_wot", "snip", "synflow"}, {"grad_norm", "grad_plain", "ntk_cond_num", "snip", "synflow"}, {"grad_norm", "nas_wot", "ntk_cond_num", "snip", "synflow"}, {"grad_plain", "nas_wot", "ntk_cond_num", "snip", "synflow"}, {"grad_norm", "grad_plain", "ntk_trace", "snip", "synflow"}, {"grad_norm", "nas_wot", "ntk_trace", "snip", "synflow"}, {"grad_plain", "nas_wot", "ntk_trace", "snip", "synflow"}, {"grad_norm", "ntk_cond_num", "ntk_trace", "snip", "synflow"}, {"grad_plain", "ntk_cond_num", "ntk_trace", "snip", "synflow"}, {"nas_wot", "ntk_cond_num", "ntk_trace", "snip", "synflow"}, {"grad_norm", "grad_plain", "ntk_trace_approx", "snip", "synflow"}, {"grad_norm", "nas_wot", "ntk_trace_approx", "snip", "synflow"}, {"grad_plain", "nas_wot", "ntk_trace_approx", "snip", "synflow"}, {"grad_norm", "ntk_cond_num", "ntk_trace_approx", "snip", "synflow"}, {"grad_plain", "ntk_cond_num", "ntk_trace_approx", "snip", "synflow"}, {"nas_wot", "ntk_cond_num", "ntk_trace_approx", "snip", "synflow"}, {"grad_norm", "ntk_trace", "ntk_trace_approx", "snip", "synflow"}, {"grad_plain", "ntk_trace", "ntk_trace_approx", "snip", "synflow"}, {"nas_wot", "ntk_trace", "ntk_trace_approx", "snip", "synflow"}, {"ntk_cond_num", "ntk_trace", "ntk_trace_approx", "snip", "synflow"}, {"grad_norm", "grad_plain", "fisher", "snip", "synflow"}, {"grad_norm", "nas_wot", "fisher", "snip", "synflow"}, {"grad_plain", "nas_wot", "fisher", "snip", "synflow"}, {"grad_norm", "ntk_cond_num", "fisher", "snip", "synflow"}, {"grad_plain", "ntk_cond_num", "fisher", "snip", "synflow"}, {"nas_wot", "ntk_cond_num", "fisher", "snip", "synflow"}, {"grad_norm", "ntk_trace", "fisher", "snip", "synflow"}, {"grad_plain", "ntk_trace", "fisher", "snip", "synflow"}, {"nas_wot", "ntk_trace", "fisher", "snip", "synflow"}, {"ntk_cond_num", "ntk_trace", "fisher", "snip", "synflow"}, {"grad_norm", "ntk_trace_approx", "fisher", "snip", "synflow"}, {"grad_plain", "ntk_trace_approx", "fisher", "snip", "synflow"}, {"nas_wot", "ntk_trace_approx", "fisher", "snip", "synflow"}, {"ntk_cond_num", "ntk_trace_approx", "fisher", "snip", "synflow"}, {"ntk_trace", "ntk_trace_approx", "fisher", "snip", "synflow"}, {"grad_norm", "grad_plain", "grasp", "snip", "synflow"}, {"grad_norm", "nas_wot", "grasp", "snip", "synflow"}, {"grad_plain", "nas_wot", "grasp", "snip", "synflow"}, {"grad_norm", "ntk_cond_num", "grasp", "snip", "synflow"}, {"grad_plain", "ntk_cond_num", "grasp", "snip", "synflow"}, {"nas_wot", "ntk_cond_num", "grasp", "snip", "synflow"}, {"grad_norm", "ntk_trace", "grasp", "snip", "synflow"}, {"grad_plain", "ntk_trace", "grasp", "snip", "synflow"}, {"nas_wot", "ntk_trace", "grasp", "snip", "synflow"}, {"ntk_cond_num", "ntk_trace", "grasp", "snip", "synflow"}, {"grad_norm", "ntk_trace_approx", "grasp", "snip", "synflow"}, {"grad_plain", "ntk_trace_approx", "grasp", "snip", "synflow"}, {"nas_wot", "ntk_trace_approx", "grasp", "snip", "synflow"}, {"ntk_cond_num", "ntk_trace_approx", "grasp", "snip", "synflow"}, {"ntk_trace", "ntk_trace_approx", "grasp", "snip", "synflow"}, {"grad_norm", "fisher", "grasp", "snip", "synflow"}, {"grad_plain", "fisher", "grasp", "snip", "synflow"}, {"nas_wot", "fisher", "grasp", "snip", "synflow"}, {"ntk_cond_num", "fisher", "grasp", "snip", "synflow"}, {"ntk_trace", "fisher", "grasp", "snip", "synflow"}, {"ntk_trace_approx", "fisher", "grasp", "snip", "synflow"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "grasp"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "fisher", "grasp"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "fisher", "grasp"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_trace", "ntk_trace_approx", "fisher", "grasp"}, {"grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "grasp"}, {"grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "grasp"}, {"grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "grasp"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "snip"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "fisher", "snip"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "fisher", "snip"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_trace", "ntk_trace_approx", "fisher", "snip"}, {"grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "snip"}, {"grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "snip"}, {"grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "snip"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "grasp", "snip"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "grasp", "snip"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_trace", "ntk_trace_approx", "grasp", "snip"}, {"grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "grasp", "snip"}, {"grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "grasp", "snip"}, {"grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "grasp", "snip"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "fisher", "grasp", "snip"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_trace", "fisher", "grasp", "snip"}, {"grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace", "fisher", "grasp", "snip"}, {"grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace", "fisher", "grasp", "snip"}, {"grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "fisher", "grasp", "snip"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_trace_approx", "fisher", "grasp", "snip"}, {"grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace_approx", "fisher", "grasp", "snip"}, {"grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "fisher", "grasp", "snip"}, {"grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "fisher", "grasp", "snip"}, {"grad_norm", "grad_plain", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", "snip"}, {"grad_norm", "nas_wot", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", "snip"}, {"grad_plain", "nas_wot", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", "snip"}, {"grad_norm", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", "snip"}, {"grad_plain", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", "snip"}, {"nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", "snip"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "synflow"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "fisher", "synflow"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "fisher", "synflow"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_trace", "ntk_trace_approx", "fisher", "synflow"}, {"grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "synflow"}, {"grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "synflow"}, {"grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "synflow"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "grasp", "synflow"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "grasp", "synflow"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_trace", "ntk_trace_approx", "grasp", "synflow"}, {"grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "grasp", "synflow"}, {"grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "grasp", "synflow"}, {"grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "grasp", "synflow"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "fisher", "grasp", "synflow"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_trace", "fisher", "grasp", "synflow"}, {"grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace", "fisher", "grasp", "synflow"}, {"grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace", "fisher", "grasp", "synflow"}, {"grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "fisher", "grasp", "synflow"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_trace_approx", "fisher", "grasp", "synflow"}, {"grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace_approx", "fisher", "grasp", "synflow"}, {"grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "fisher", "grasp", "synflow"}, {"grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "fisher", "grasp", "synflow"}, {"grad_norm", "grad_plain", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", "synflow"}, {"grad_norm", "nas_wot", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", "synflow"}, {"grad_plain", "nas_wot", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", "synflow"}, {"grad_norm", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", "synflow"}, {"grad_plain", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", "synflow"}, {"nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", "synflow"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "snip", "synflow"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "snip", "synflow"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_trace", "ntk_trace_approx", "snip", "synflow"}, {"grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "snip", "synflow"}, {"grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "snip", "synflow"}, {"grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "snip", "synflow"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "fisher", "snip", "synflow"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_trace", "fisher", "snip", "synflow"}, {"grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace", "fisher", "snip", "synflow"}, {"grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace", "fisher", "snip", "synflow"}, {"grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "fisher", "snip", "synflow"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_trace_approx", "fisher", "snip", "synflow"}, {"grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace_approx", "fisher", "snip", "synflow"}, {"grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "fisher", "snip", "synflow"}, {"grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "fisher", "snip", "synflow"}, {"grad_norm", "grad_plain", "ntk_trace", "ntk_trace_approx", "fisher", "snip", "synflow"}, {"grad_norm", "nas_wot", "ntk_trace", "ntk_trace_approx", "fisher", "snip", "synflow"}, {"grad_plain", "nas_wot", "ntk_trace", "ntk_trace_approx", "fisher", "snip", "synflow"}, {"grad_norm", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "snip", "synflow"}, {"grad_plain", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "snip", "synflow"}, {"nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "snip", "synflow"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "grasp", "snip", "synflow"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_trace", "grasp", "snip", "synflow"}, {"grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace", "grasp", "snip", "synflow"}, {"grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace", "grasp", "snip", "synflow"}, {"grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "grasp", "snip", "synflow"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_trace_approx", "grasp", "snip", "synflow"}, {"grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace_approx", "grasp", "snip", "synflow"}, {"grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "grasp", "snip", "synflow"}, {"grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "grasp", "snip", "synflow"}, {"grad_norm", "grad_plain", "ntk_trace", "ntk_trace_approx", "grasp", "snip", "synflow"}, {"grad_norm", "nas_wot", "ntk_trace", "ntk_trace_approx", "grasp", "snip", "synflow"}, {"grad_plain", "nas_wot", "ntk_trace", "ntk_trace_approx", "grasp", "snip", "synflow"}, {"grad_norm", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "grasp", "snip", "synflow"}, {"grad_plain", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "grasp", "snip", "synflow"}, {"nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "grasp", "snip", "synflow"}, {"grad_norm", "grad_plain", "nas_wot", "fisher", "grasp", "snip", "synflow"}, {"grad_norm", "grad_plain", "ntk_cond_num", "fisher", "grasp", "snip", "synflow"}, {"grad_norm", "nas_wot", "ntk_cond_num", "fisher", "grasp", "snip", "synflow"}, {"grad_plain", "nas_wot", "ntk_cond_num", "fisher", "grasp", "snip", "synflow"}, {"grad_norm", "grad_plain", "ntk_trace", "fisher", "grasp", "snip", "synflow"}, {"grad_norm", "nas_wot", "ntk_trace", "fisher", "grasp", "snip", "synflow"}, {"grad_plain", "nas_wot", "ntk_trace", "fisher", "grasp", "snip", "synflow"}, {"grad_norm", "ntk_cond_num", "ntk_trace", "fisher", "grasp", "snip", "synflow"}, {"grad_plain", "ntk_cond_num", "ntk_trace", "fisher", "grasp", "snip", "synflow"}, {"nas_wot", "ntk_cond_num", "ntk_trace", "fisher", "grasp", "snip", "synflow"}, {"grad_norm", "grad_plain", "ntk_trace_approx", "fisher", "grasp", "snip", "synflow"}, {"grad_norm", "nas_wot", "ntk_trace_approx", "fisher", "grasp", "snip", "synflow"}, {"grad_plain", "nas_wot", "ntk_trace_approx", "fisher", "grasp", "snip", "synflow"}, {"grad_norm", "ntk_cond_num", "ntk_trace_approx", "fisher", "grasp", "snip", "synflow"}, {"grad_plain", "ntk_cond_num", "ntk_trace_approx", "fisher", "grasp", "snip", "synflow"}, {"nas_wot", "ntk_cond_num", "ntk_trace_approx", "fisher", "grasp", "snip", "synflow"}, {"grad_norm", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", "snip", "synflow"}, {"grad_plain", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", "snip", "synflow"}, {"nas_wot", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", "snip", "synflow"}, {"ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", "snip", "synflow"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", "snip"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", "synflow"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "snip", "synflow"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "grasp", "snip", "synflow"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "fisher", "grasp", "snip", "synflow"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "fisher", "grasp", "snip", "synflow"}, {"grad_norm", "grad_plain", "nas_wot", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", "snip", "synflow"}, {"grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", "snip", "synflow"}, {"grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", "snip", "synflow"}, {"grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", "snip", "synflow"}}
	totalCores := 8
	runtime.GOMAXPROCS(totalCores)

	var wg sync.WaitGroup
	wg.Add(totalCores)

	start0 := time.Now()

	for i := 0; i < totalCores; i++ {

		fileName := fmt.Sprintf("/Users/kevin/project_python/Fast-AutoNAS/main/others/partition-%d", i)
		archPartitonMapper := readJsonArchIds(fileName)
		arch_combinations := getKeys2(archPartitonMapper)

		fmt.Printf("Run one thread computes %d archs \n", len(arch_combinations))
		go func(archs []string, iteIndexM int) {
			defer wg.Done()

			checkAllVote(data201UnionBest,
				"201union_best",
				all_vote_metrics,
				fmt.Sprintf("file-%d", iteIndexM),
				archs)
		}(arch_combinations, i)
	}

	fmt.Println("main thread wait here...")
	wg.Wait()
	fmt.Printf("201union, execution time %s\n", time.Since(start0))
}

func measureSelectedVote(data map[string]archInfo) {

}

func readJsonArchIds(fileRes string) map[string]int {
	fileContent, err := os.Open(fileRes)
	if err != nil {
		log.Fatal(err)
	}
	defer fileContent.Close()
	byteResult, _ := ioutil.ReadAll(fileContent)
	var res map[string]int
	err = json.Unmarshal(byteResult, &res)
	if err != nil {
		fmt.Println(err)
	}
	return res
}

func main() {

	data_201_bn := readJson("./Logs/cifar100_15000/201_15k_c100_128_BN_str.json")
	//data_201_no_bn := readJson("/opt/Fast-AutoNAS/main/201_15k_c10_128_noBN_str.json")
	//data201UnionBest := readJson("/opt/Fast-AutoNAS/main/201_15k_c10_128_union_best_str.json")

	measureAllVoteMetrics(data_201_bn)

}
