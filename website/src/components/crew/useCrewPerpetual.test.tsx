import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";

const H = vi.hoisted(() => ({
  members: vi.fn(),
  autonudgeList: vi.fn(),
}));

vi.mock("../../api/client", () => ({
  api: {
    members: H.members,
    autonudgeList: H.autonudgeList,
  },
}));
import { AUTONUDGE_LOOPS_QUERY_KEY } from "../autoNudgeLoop";

import { MEMBERS_ROSTER_QUERY_KEY } from "../../api/membersQuery";
import {
  PERPETUAL_DEFAULT_BANNER,
  PERPETUAL_DEFAULT_IDLE_SECS,
  PERPETUAL_DEFAULT_INSTRUCTION,
  isDefaultPerpetualBrief,
  perpetualBriefText,
  perpetualIdleSecs,
  useCrewPerpetual,
  wakesPerDay,
} from "./useCrewPerpetual";

describe("perpetualBriefText", () => {
  // The instruction row on the Crewmates page shows a loop's banner (else its
  // message). For a loop the owner's switch armed, both are the switch's own
  // default brief, which only restates the block's title -- hidden. Anything
  // the crewmate wrote itself is kept, whichever field carries it.
  it("treats exactly the default banner, the default instruction and the feature name as default", () => {
    expect(isDefaultPerpetualBrief(PERPETUAL_DEFAULT_BANNER)).toBe(true);
    expect(isDefaultPerpetualBrief(PERPETUAL_DEFAULT_INSTRUCTION)).toBe(true);
    expect(isDefaultPerpetualBrief("Perpetual mode")).toBe(true);
    expect(isDefaultPerpetualBrief("")).toBe(true);
    expect(isDefaultPerpetualBrief(undefined)).toBe(true);
    // A word changed is a brief of its own.
    expect(isDefaultPerpetualBrief(`${PERPETUAL_DEFAULT_BANNER}.`)).toBe(false);
    expect(isDefaultPerpetualBrief("Perpetual mode: watch the queue")).toBe(
      false,
    );
  });

  it("hides the owner-armed default, keeps a custom banner, and falls through to a custom message under the default banner", () => {
    expect(
      perpetualBriefText({
        banner: PERPETUAL_DEFAULT_BANNER,
        message: PERPETUAL_DEFAULT_INSTRUCTION,
      }),
    ).toBe("");
    expect(perpetualBriefText({ banner: "", message: "" })).toBe("");
    expect(
      perpetualBriefText({
        banner: "watching PR #123",
        message: PERPETUAL_DEFAULT_INSTRUCTION,
      }),
    ).toBe("watching PR #123");
    expect(
      perpetualBriefText({
        banner: PERPETUAL_DEFAULT_BANNER,
        message: "Triage the inbox.",
      }),
    ).toBe("Triage the inbox.");
    expect(
      perpetualBriefText({
        banner: "",
        message: "Patrol the queue.\nSecond line",
      }),
    ).toBe("Patrol the queue.\nSecond line");
  });
});

describe("perpetualIdleSecs", () => {
  // The interval the switch states its facts with: the crewmate's own once it
  // has a record, and the interval an arm starts on until then -- so the
  // first-wake timing and the cost estimate are never blank and never zero.
  it("prefers the record, and falls back to the interval an arm starts on", () => {
    expect(perpetualIdleSecs({ idle_secs: 900 })).toBe(900);
    expect(perpetualIdleSecs(undefined)).toBe(PERPETUAL_DEFAULT_IDLE_SECS);
    expect(perpetualIdleSecs(null)).toBe(PERPETUAL_DEFAULT_IDLE_SECS);
    expect(perpetualIdleSecs({})).toBe(PERPETUAL_DEFAULT_IDLE_SECS);
    // A torn or pre-field record: 0 is not an interval, and rendering it would
    // read as "wakes constantly".
    expect(perpetualIdleSecs({ idle_secs: 0 })).toBe(
      PERPETUAL_DEFAULT_IDLE_SECS,
    );
    expect(perpetualIdleSecs({ idle_secs: -60 })).toBe(
      PERPETUAL_DEFAULT_IDLE_SECS,
    );
  });
});

describe("wakesPerDay", () => {
  it("reads a whole number of wakes a day at the common intervals", () => {
    expect(wakesPerDay(PERPETUAL_DEFAULT_IDLE_SECS)).toBe(24);
    expect(wakesPerDay(900)).toBe(96);
    expect(wakesPerDay(300)).toBe(288);
    expect(wakesPerDay(43_200)).toBe(2);
  });

  it("keeps one decimal where a whole number would round the estimate away", () => {
    // An interval at or past a day must not collapse to "0" and claim the
    // crewmate never wakes, and 6 h is 4 a day, not "4.0".
    expect(wakesPerDay(86_400)).toBe(1);
    expect(wakesPerDay(172_800)).toBe(0.5);
    expect(wakesPerDay(21_600)).toBe(4);
    // Above ten a day the decimal carries nothing: 90 m is "16", and 7 m is a
    // whole "206" rather than "205.7".
    expect(wakesPerDay(5_400)).toBe(16);
    expect(wakesPerDay(420)).toBe(206);
  });

  it("reads a missing or nonsense interval as the default arm, never as zero", () => {
    expect(wakesPerDay(0)).toBe(24);
    expect(wakesPerDay(-1)).toBe(24);
    expect(wakesPerDay(Number.NaN)).toBe(24);
    expect(wakesPerDay(Number.POSITIVE_INFINITY)).toBe(24);
  });
});

describe("useCrewPerpetual", () => {
  // `H` is hoisted once for the module, so without a reset each test would
  // inherit the previous test's call history and seeded implementations. The
  // retry test counts calls from its own baseline (1, then +1), which is only
  // deterministic when the history starts empty.
  beforeEach(() => {
    H.members.mockReset();
    H.autonudgeList.mockReset();
  });

  it("skips the initial roster refetch, then refreshes after a registry update", async () => {
    H.members.mockResolvedValue({
      members: [
        {
          name: "Radar",
          slug: "radar",
          slot_key: "member-radar",
          perpetual: "on",
        },
      ],
    });
    H.autonudgeList.mockResolvedValue({ enabled: true, loops: [] });
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    const invalidate = vi.spyOn(queryClient, "invalidateQueries");
    const wrapper = ({ children }: { children: ReactNode }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );

    renderHook(() => useCrewPerpetual("Radar", { poll: false }), { wrapper });

    await waitFor(() => expect(H.autonudgeList).toHaveBeenCalledTimes(1));
    expect(invalidate).not.toHaveBeenCalledWith({
      queryKey: MEMBERS_ROSTER_QUERY_KEY,
    });
    expect(H.members).toHaveBeenCalledTimes(1);

    await queryClient.refetchQueries({ queryKey: AUTONUDGE_LOOPS_QUERY_KEY });

    await waitFor(() =>
      expect(invalidate).toHaveBeenCalledWith({
        queryKey: MEMBERS_ROSTER_QUERY_KEY,
      }),
    );
    expect(H.members).toHaveBeenCalledTimes(2);
  });

  it("retry re-asks both reads and clears a failed reading once they answer", async () => {
    // The reading is assembled from two queries. When one never answered, the
    // host's notice offers `retry`, which must refetch BOTH -- not only the one
    // that failed -- so the reading settles on two fresh answers.
    H.members.mockResolvedValue({
      members: [
        {
          name: "Radar",
          slug: "radar",
          slot_key: "member-radar",
          perpetual: "on",
        },
      ],
    });
    H.autonudgeList
      .mockRejectedValueOnce(new Error("boom"))
      .mockResolvedValue({ enabled: true, loops: [] });
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    const wrapper = ({ children }: { children: ReactNode }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );

    const { result } = renderHook(
      () => useCrewPerpetual("Radar", { poll: false }),
      { wrapper },
    );

    await waitFor(() => expect(result.current.loaded).toBe(true));
    expect(result.current.failed).toBe(true);
    expect(H.autonudgeList).toHaveBeenCalledTimes(1);
    expect(H.members).toHaveBeenCalledTimes(1);

    result.current.retry();

    await waitFor(() => expect(H.autonudgeList).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(H.members).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(result.current.failed).toBe(false));
    expect(result.current.loaded).toBe(true);
    expect(result.current.state).toBe("on");
  });

  it("does not label a full self-arm row as a structured monitor", async () => {
    H.members.mockResolvedValue({
      members: [
        {
          name: "Radar",
          slug: "radar",
          slot_key: "member-radar",
          perpetual: "none",
        },
      ],
    });
    H.autonudgeList.mockResolvedValue({
      enabled: true,
      loops: [
        {
          id: "finite-self-arm",
          slot_key: "member-radar",
          message: "Check the queue.",
          active: true,
          idle_secs: 300,
          max_cycles: 24,
          max_runtime_secs: 3600,
        },
      ],
    });
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    const wrapper = ({ children }: { children: ReactNode }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );

    const { result } = renderHook(() => useCrewPerpetual("Radar", { poll: false }), {
      wrapper,
    });

    await waitFor(() => expect(result.current.loaded).toBe(true));
    expect(result.current.state).toBe("none");
    expect(result.current.monitor).toBe(false);
  });
});
